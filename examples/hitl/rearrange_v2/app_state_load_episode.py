#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

from app_data import AppData
from app_state_base import AppStateBase
from app_states import (
    create_app_state_cancel_session,
    create_app_state_end_session,
    create_app_state_start_screen,
)
from session import Session
from util import get_top_down_view

from habitat_hitl.app_states.app_service import AppService
from habitat_hitl.core.user_mask import Mask


class AppStateLoadEpisode(AppStateBase):
    """
    Load an episode.
    A loading screen is shown while the content loads.
    * If a next episode exists, launch RearrangeV2.
    * If all episodes are done, end session.
    * If any user disconnects, cancel the session.
    """

    def __init__(
        self, app_service: AppService, app_data: AppData, session: Session
    ):
        super().__init__(app_service, app_data)
        self._session = session
        self._loading = True
        self._session_ended = False
        self._frame_number = 0
        self._save_keyframes = False

    def get_next_state(self) -> Optional[AppStateBase]:
        if self._cancel:
            return create_app_state_cancel_session(
                self._app_service,
                self._app_data,
                self._session,
                "User disconnected.",
            )
        if self._session_ended:
            return create_app_state_end_session(
                self._app_service, self._app_data, self._session
            )
        # When all clients finish loading, show the start screen.
        if not self._loading:
            return create_app_state_start_screen(
                self._app_service, self._app_data, self._session
            )
        return None

    def sim_update(self, dt: float, post_sim_update_dict):
        self._status_message("Loading...")

        # Skip a frame so that the status message reaches the client before the server loads the scene and blocks.
        if self._frame_number == 1:
            self._increment_episode()
        # Once the scene loaded, show a top-down view.
        elif self._frame_number > 1:
            cam_matrix = get_top_down_view(self._app_service.sim)
            post_sim_update_dict["cam_transform"] = cam_matrix
            self._app_service._client_message_manager.update_camera_transform(
                cam_matrix, destination_mask=Mask.ALL
            )
            # Wait for clients to signal that content finished loading on their end.
            # HACK: The server isn't immediately aware that clients are loading. For now, we simply skip some frames.
            # TODO: Use the keyframe ID from 'ClientMessageManager.set_server_keyframe_id()' to find the when the loading state is up-to-date.
            if self._frame_number > 20:
                any_client_loading = False
                for user_index in range(self._app_data.max_user_count):
                    if self._app_service.remote_client_state._client_loading[
                        user_index
                    ]:
                        any_client_loading = True
                        break
                if not any_client_loading:
                    self._loading = False

        self._frame_number += 1

    def _increment_episode(self):
        session = self._session
        assert session.episode_ids is not None
        if session.current_episode_index < len(session.episode_ids):
            self._set_episode(session.current_episode_index)
            session.current_episode_index += 1
        else:
            self._session_ended = True

    def _set_episode(self, episode_index: int):
        session = self._session

        # Set the ID of the next episode to play in lab.
        next_episode_id = session.episode_ids[episode_index]
        print(f"Next episode index: {next_episode_id}.")
        try:
            next_episode_index = int(next_episode_id)
            self._app_service.episode_helper.set_next_episode_by_index(
                next_episode_index
            )
        except Exception as e:
            print(f"ERROR: Invalid episode index {next_episode_id}. {e}")
            print("Loading episode index 0.")
            self._app_service.episode_helper.set_next_episode_by_index(0)

        # Once an episode ID has been set, lab needs to be reset to load the episode.
        self._app_service.end_episode(do_reset=True)

        # Signal the clients that the scene has changed.
        client_message_manager = self._app_service.client_message_manager
        if client_message_manager:
            client_message_manager.signal_scene_change(Mask.ALL)

        # Save a keyframe. This propagates the new content to the clients, initiating client-side loading.
        # Beware that the client "loading" state won't immediately be visible to the server.
        self._app_service.sim.gfx_replay_manager.save_keyframe()

#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os.path as osp
from typing import Dict

import numpy as np
import pandas as pd

from habitat.core.registry import registry
from habitat.datasets.ovmm.ovmm_dataset import OVMMEpisode
from habitat.tasks.rearrange.sub_tasks.nav_to_obj_task import DynNavRLEnv


@registry.register_task(name="OVMMNavToObjTask-v0")
class OVMMDynNavRLEnv(DynNavRLEnv):
    def __init__(self, *args, config, dataset=None, **kwargs):
        super().__init__(
            config=config,
            *args,
            dataset=dataset,
            **kwargs,
        )
        self._receptacle_semantic_ids: Dict[int, int] = {}
        self._receptacle_categories: Dict[str, str] = {}
        self._object_semantic_ids: Dict[int, int] = {}
        self._object_categories: Dict[str, str] = {}
        self._other_object_semantic_ids: Dict[int, int] = {}
        self._other_object_categories: Dict[str, str] = {}
        self._recep_category_to_recep_category_id = (
            dataset.recep_category_to_recep_category_id
        )
        self._obj_category_to_obj_category_id = (
            dataset.obj_category_to_obj_category_id
        )
        self._other_obj_category_to_other_obj_category_id = (
            dataset.other_obj_category_to_other_obj_category_id
        )
        self._loaded_receptacle_categories = False
        if config.receptacle_categories_file is not None and osp.exists(
            config.receptacle_categories_file
        ):
            with open(config.receptacle_categories_file) as f:
                for line in f.readlines():
                    name, category = line.strip().split(",")
                    self._receptacle_categories[name] = category
            self._loaded_receptacle_categories = True

        self._loaded_object_categories = False
        if config.object_categories_file is not None and osp.exists(config.object_categories_file):
            with open(config.object_categories_file) as f:
                for line in f.readlines():
                    name, category = line.strip().split(",")
                    self._object_categories[name] = category
            self._loaded_object_categories = True

        self.load_other_object_categories_file(
            config.other_object_categories_file
        )

    def load_other_object_categories_file(
        self,
        other_object_categories_file: str,
    ):
        """
        Load Receptacle category mapping file to generate a dictionary of Receptacle.unique_names to their category.

        """
        self._loaded_other_object_categories = False
        if other_object_categories_file is not None and osp.exists(other_object_categories_file):
            df = pd.read_csv(other_object_categories_file)
            name_key = "id" if "id" in df else "name"
            category_key = (
                "main_category" if "main_category" in df else "clean_category"
            )

            df["category"] = (
                df[category_key]
                .fillna("")
                .apply(lambda x: x.replace(" ", "_").split(".")[0])
            )
            self._other_object_categories = dict(zip(df[name_key], df["category"]))
            # remove objects that are in object categories or receptacle categories
            self._other_object_categories = {
                k: v
                for k, v in self._other_object_categories.items()
                if k not in self._object_categories and k not in self._receptacle_categories
            }
            self._loaded_other_object_categories = True


    @property
    def receptacle_semantic_ids(self):
        return self._receptacle_semantic_ids

    @property
    def loaded_receptacle_categories(self):
        return self._loaded_receptacle_categories

    @property
    def object_semantic_ids(self):
        return self._object_semantic_ids

    @property
    def loaded_object_categories(self):
        return self._loaded_object_categories

    @property
    def other_object_semantic_ids(self):
        return self._other_object_semantic_ids

    @property
    def loaded_other_object_categories(self):
        return self._loaded_other_object_categories

    def reset(self, episode: OVMMEpisode):
        self._receptacle_semantic_ids = {}
        self._cache_receptacles()
        self._object_semantic_ids = {}
        self._cache_objects()
        self._other_object_semantic_ids = {}
        self._cache_other_objects()
        obs = super().reset(episode)
        self._nav_to_obj_goal = np.stack(
            [
                view_point.agent_state.position
                for goal in episode.candidate_objects
                for view_point in goal.view_points
            ],
            axis=0,
        )
        return obs

    def _cache_receptacles(self):
        # TODO: potentially this is slow, get receptacle list from episode instead
        rom = self._sim.get_rigid_object_manager()
        for obj_handle in rom.get_object_handles():
            obj = rom.get_object_by_handle(obj_handle)
            user_attr_keys = obj.user_attributes.get_subconfig_keys()
            if any(key.startswith("receptacle_") for key in user_attr_keys):
                obj_name = osp.basename(obj.creation_attributes.handle).split(
                    ".", 1
                )[0]
                category = self._receptacle_categories.get(obj_name)
                if (
                    category is None
                    or category
                    not in self._recep_category_to_recep_category_id
                ):
                    continue
                category_id = self._recep_category_to_recep_category_id[
                    category
                ]
                self._receptacle_semantic_ids[obj.object_id] = category_id + 1

    def _cache_objects(self):
        rom = self._sim.get_rigid_object_manager()
        for scene_obj_id in self._sim.scene_obj_ids:
            # get the handle
            handle = rom.get_object_by_id(scene_obj_id).handle
            category = self._object_categories.get(handle[:-6])
            if category is None or category not in self._obj_category_to_obj_category_id:
                continue
            category_id = self._obj_category_to_obj_category_id[
                category
            ]
            self._object_semantic_ids[scene_obj_id] = category_id + 1

    def _cache_other_objects(self):
        rom = self._sim.get_rigid_object_manager()
        for obj_handle in rom.get_object_handles():
            obj = rom.get_object_by_handle(obj_handle)

            # confirm object is not a receptacle
            user_attr_keys = obj.user_attributes.get_subconfig_keys()
            if any(key.startswith("receptacle_") for key in user_attr_keys):
                continue

            # confirm object is not a pickupable object
            obj_name = obj_handle[:-6]
            category = self._object_categories.get(obj_name)
            if category in self._obj_category_to_obj_category_id:
                continue

            category = self._other_object_categories.get(obj_name)

            if (
                category is None
                or category
                not in self._other_obj_category_to_other_obj_category_id
            ):
                continue
            category_id = self._other_obj_category_to_other_obj_category_id[
                category
            ]
            self._other_object_semantic_ids[obj.object_id] = category_id + 1

    def _generate_nav_to_pos(
        self, episode, start_hold_obj_idx=None, force_idx=None
    ):
        # learn nav to pick skill if not holding object currently
        if start_hold_obj_idx is None:
            # starting positions of candidate objects
            all_pos = np.stack(
                [
                    view_point.agent_state.position
                    for goal in episode.candidate_objects
                    for view_point in goal.view_points
                ],
                axis=0,
            )
            if force_idx is not None:
                raise NotImplementedError
        else:
            # positions of candidate goal receptacles
            all_pos = np.stack(
                [
                    view_point.agent_state.position
                    for goal in episode.candidate_goal_receps
                    for view_point in goal.view_points
                ],
                axis=0,
            )

        return all_pos
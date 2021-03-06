# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.
import os
import json
import numpy as np
from collections.abc import Iterable

import grid2op
from l2rpn_baselines.utils.baseDeepQ import BaseDeepQ


class NNParam(object):
    """
    This class provides an easy way to save and restore, as json, the shape of your neural networks
    (number of layers, non linearities, size of each layers etc.)

    It is recommended to overload this class for each specific model.

    .. warning::
        This baseline recodes entire the RL training procedure. You can use it if you
        want to have a deeper look at Deep Q Learning algorithm and a possible (non 
        optimized, slow, etc. implementation ).
        
        For a much better implementation, you can reuse the code of "PPO_RLLIB" 
        or the "PPO_SB3" baseline.
        
        Prefer to use the :class:`GymAgent` class and the :class:`GymEnvWithHeuristics`
        classes to train agent interacting with grid2op and fully compatible
        with gym framework.	
        
    Attributes
    ----------

    nn_class: :class:`l2rpn_baselines.BaseDeepQ`
        The neural network class that will be created with each call of :func:`l2rpn_baselines.make_nn`

    observation_size: ``int``
        The size of the observation space.

    action_size: ``int``
        The size of the action space.

    sizes: ``list``
        A list of integer, each will represent the number of hidden units. The number of hidden layer is given by
        the size / length of this list.

    activs: ``list``
        List of activation functions (given as string). It should have the same length as the :attr:`NNParam.sizes`.
        This function should be name of keras activation function.

    list_attr_obs: ``list``
        List of the attributes that will be used from the observation and concatenated to be fed to the neural network.

    """

    _int_attr = ["action_size", "observation_size"]
    _float_attr = []
    _str_attr = []
    _list_float = []
    _list_str = ["activs", "list_attr_obs"]
    _list_int = ["sizes"]
    nn_class = BaseDeepQ

    def __init__(self,
                 action_size,
                 observation_size,
                 sizes,
                 activs,
                 list_attr_obs,
                 ):
        self.observation_size = observation_size
        self.action_size = action_size
        self.sizes = [int(el) for el in sizes]
        self.activs = [str(el) for el in activs]
        if len(self.sizes) != len(self.activs):
            raise RuntimeError("\"sizes\" and \"activs\" lists have not the same size. It's not clear how many layers "
                               "you want your neural network to have.")
        self.list_attr_obs = [str(el) for el in list_attr_obs]

    @classmethod
    def get_path_model(cls, path, name=None):
        """get the path at which the model will be saved"""
        return cls.nn_class.get_path_model(path, name=name)

    def make_nn(self, training_param):
        """build the appropriate BaseDeepQ"""
        res = self.nn_class(self, training_param)
        return res

    @staticmethod
    def get_obs_size(env, list_attr_name):
        """get the size of the flatten observation"""
        res = 0
        for obs_attr_name in list_attr_name:
            beg_, end_, dtype_ = env.observation_space.get_indx_extract(obs_attr_name)
            res += end_ - beg_  # no "+1" needed because "end_" is exclude by python convention
        return res

    def get_obs_attr(self):
        """get the names of the observation attributes that will be extracted """
        return self.list_attr_obs

    # utilitaries, do not change
    def to_dict(self):
        """convert this instance to a dictionnary"""
        # TODO copy and paste from TrainingParam
        res = {}
        for attr_nm in self._int_attr:
            tmp = getattr(self, attr_nm)
            if tmp is not None:
                res[attr_nm] = int(tmp)
            else:
                res[attr_nm] = None
        for attr_nm in self._float_attr:
            tmp = getattr(self, attr_nm)
            if tmp is not None:
                res[attr_nm] = float(tmp)
            else:
                res[attr_nm] = None
        for attr_nm in self._str_attr:
            tmp = getattr(self, attr_nm)
            if tmp is not None:
                res[attr_nm] = str(tmp)
            else:
                res[attr_nm] = None

        for attr_nm in self._list_float:
            tmp = getattr(self, attr_nm)
            res[attr_nm] = self._convert_list_to_json(tmp, float)
        for attr_nm in self._list_int:
            tmp = getattr(self, attr_nm)
            res[attr_nm] = self._convert_list_to_json(tmp, int)
        for attr_nm in self._list_str:
            tmp = getattr(self, attr_nm)
            res[attr_nm] = self._convert_list_to_json(tmp, str)
        return res

    @classmethod
    def _convert_list_to_json(cls, obj, type_):
        if isinstance(obj, type_):
            res = obj
        elif isinstance(obj, np.ndarray):
            if len(obj.shape) == 1:
                res = [type_(el) for el in obj]
            else:
                res = [cls._convert_list_to_json(el, type_) for el in obj]
        elif isinstance(obj, Iterable):
            res = [cls._convert_list_to_json(el, type_) for el in obj]
        else:
            res = type_(obj)
        return res

    @classmethod
    def _attr_from_json(cls, json, type_):
        if isinstance(json, type_):
            res = json
        elif isinstance(json, list):
            res = [cls._convert_list_to_json(obj=el, type_=type_) for el in json]
        else:
            res = type_(json)
        return res

    @classmethod
    def from_dict(cls, tmp):
        """load from a dictionnary"""
        # TODO copy and paste from TrainingParam (more or less)
        cls_as_dict = {}
        for attr_nm in cls._int_attr:
            if attr_nm in tmp:
                tmp_ = tmp[attr_nm]
                if tmp_ is not None:
                    cls_as_dict[attr_nm] = int(tmp_)
                else:
                    cls_as_dict[attr_nm] = None

        for attr_nm in cls._float_attr:
            if attr_nm in tmp:
                tmp_ = tmp[attr_nm]
                if tmp_ is not None:
                    cls_as_dict[attr_nm] = float(tmp_)
                else:
                    cls_as_dict[attr_nm] = None

        for attr_nm in cls._str_attr:
            if attr_nm in tmp:
                tmp_ = tmp[attr_nm]
                if tmp_ is not None:
                    cls_as_dict[attr_nm] = str(tmp_)
                else:
                    cls_as_dict[attr_nm] = None

        for attr_nm in cls._list_float:
            if attr_nm in tmp:
                cls_as_dict[attr_nm] = cls._attr_from_json(tmp[attr_nm], float)
        for attr_nm in cls._list_int:
            if attr_nm in tmp:
                cls_as_dict[attr_nm] = cls._attr_from_json(tmp[attr_nm], int)
        for attr_nm in cls._list_str:
            if attr_nm in tmp:
                cls_as_dict[attr_nm] = cls._attr_from_json(tmp[attr_nm], str)

        res = cls(**cls_as_dict)
        return res

    @classmethod
    def from_json(cls, json_path):
        """load from a json file"""
        # TODO copy and paste from TrainingParam
        if not os.path.exists(json_path):
            raise FileNotFoundError("No path are located at \"{}\"".format(json_path))
        with open(json_path, "r") as f:
            dict_ = json.load(f)
        return cls.from_dict(dict_)

    def save_as_json(self, path, name=None):
        """save as a json file"""
        # TODO copy and paste from TrainingParam
        res = self.to_dict()
        if name is None:
            name = "neural_net_parameters.json"
        if not os.path.exists(path):
            raise RuntimeError("Directory \"{}\" not found to save the NN parameters".format(path))
        if not os.path.isdir(path):
            raise NotADirectoryError("\"{}\" should be a directory".format(path))
        path_out = os.path.join(path, name)
        with open(path_out, "w", encoding="utf-8") as f:
            json.dump(res, fp=f, indent=4, sort_keys=True)

    def center_reduce(self, env):
        """currently not implemented for this class, "coming soon" as we might say"""
        # TODO see TestLeapNet for this feature
        self._center_reduce_vect(env.get_obs(), "x")

    def _get_adds_mults_from_name(self, obs, attr_nm):
        if attr_nm in ["prod_p"]:
            add_tmp = np.array([-0.5 * (pmax + pmin) for pmin, pmax in zip(obs.gen_pmin, obs.gen_pmax)])
            mult_tmp = np.array([1. / max((pmax - pmin), 0.) for pmin, pmax in zip(obs.gen_pmin, obs.gen_pmax)])
        elif attr_nm in ["prod_q"]:
            add_tmp = 0.
            mult_tmp = np.array([1. / max(abs(val), 1.0) for val in obs.prod_q])
        elif attr_nm in ["load_p", "load_q"]:
            add_tmp = np.array([-val for val in getattr(obs, attr_nm)])
            mult_tmp = 0.5
        elif attr_nm in ["load_v", "prod_v", "v_or", "v_ex"]:
            add_tmp = 0.
            mult_tmp = np.array([1. / val for val in getattr(obs, attr_nm)])
        elif attr_nm == "hour_of_day":
            add_tmp = -12.
            mult_tmp = 1.0 / 12
        elif attr_nm == "minute_of_hour":
            add_tmp = -30.
            mult_tmp = 1.0 / 30
        elif attr_nm == "day_of_week":
            add_tmp = -4.
            mult_tmp = 1.0 / 4
        elif attr_nm == "day":
            add_tmp = -15.
            mult_tmp = 1.0 / 15.
        elif attr_nm in ["target_dispatch", "actual_dispatch"]:
            add_tmp = 0.
            mult_tmp = np.array([1. / (pmax - pmin) for pmin, pmax in zip(obs.gen_pmin, obs.gen_pmax)])
        elif attr_nm in ["a_or", "a_ex", "p_or", "p_ex", "q_or", "q_ex"]:
            add_tmp = 0.
            mult_tmp = np.array([1.0 / max(val, 1.0) for val in getattr(obs, attr_nm)])
        else:
            add_tmp = 0.
            mult_tmp = 1.0
        return add_tmp, mult_tmp

    def _center_reduce_vect(self, obs, nn_part):
        """
        compute the xxxx_adds and xxxx_mults for one part of the neural network called nn_part,
        depending on what attribute of the observation is extracted
        """
        if not isinstance(obs, grid2op.Observation.BaseObservation):
            # in multi processing i receive a set of observation there so i might need
            # to extract only the first one
            obs = obs[0]

        li_attr_obs = getattr(self, "list_attr_obs_{}".format(nn_part))
        adds = []
        mults = []
        for attr_nm in li_attr_obs:
            add_tmp, mult_tmp = self._get_adds_mults_from_name(obs, attr_nm)
            mults.append(mult_tmp)
            adds.append(add_tmp)
        setattr(self, "{}_adds".format(nn_part), adds)
        setattr(self, "{}_mults".format(nn_part), mults)

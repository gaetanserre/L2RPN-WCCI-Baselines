# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.


from l2rpn_baselines.utils import DeepQAgent

DEFAULT_NAME = "LeapNetEncoded"


class LeapNetEncoded(DeepQAgent):
    """
    Inheriting from :class:`l2rpn_baselines.utils.deepQAgent.DeepQAgent` 
    this class implements the  particular agent used for the
    Double Duelling Deep Q network baseline, with the particularity 
    that the Q network is encoded with a leap net.

    It does nothing in particular.
    
    .. warning::
        This baseline recodes entire the RL training procedure. You can use it if you
        want to have a deeper look at Deep Q Learning algorithm and a possible (non 
        optimized, slow, etc. implementation ).
        
        For a much better implementation, you can reuse the code of "PPO_RLLIB" 
        or the "PPO_SB3" baseline.
        
    """
    pass

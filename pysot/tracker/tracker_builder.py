# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pysot.core.config import cfg
from pysot.tracker.siamrpn_tracker import SiamRPNTracker
from pysot.tracker.siamrpn_attack_template import SiamRPNAttackTemplate
from pysot.tracker.siamrpn_attack_template_search import SiamRPNAttackTemplateSearch
from pysot.tracker.siammask_tracker import SiamMaskTracker
from pysot.tracker.siamrpnlt_tracker import SiamRPNLTTracker
from pysot.tracker.siammask_tracker_attack import SiamMaskTrackerAttack
from pysot.tracker.siamrpn_attack_template_hpc import SiamRPNAttackTemplateHPC
from pysot.tracker.siamrpn_adv import SiamRPNAdv
from pysot.tracker.siamrpn_visualize import SiamRPNVisualize
from pysot.tracker.siamrpn_attack_feature import SiamRPNAttackFeature

TRACKS = {
    'SiamRPNTracker': SiamRPNTracker,
    'SiamMaskTracker': SiamMaskTracker,
    'SiamMaskTrackerAttack': SiamMaskTrackerAttack,
    'SiamRPNLTTracker': SiamRPNLTTracker,
    'SiamRPNAttackTemplate': SiamRPNAttackTemplate,
    'SiamRPNAttackTemplateSearch': SiamRPNAttackTemplateSearch,
    'SiamRPNAttackTemplateHPC': SiamRPNAttackTemplateHPC,
    'SiamRPNAdv': SiamRPNAdv,
    'SiamRPNVisualize': SiamRPNVisualize,
    'SiamRPNAttackFeature': SiamRPNAttackFeature,
}


def build_tracker(model):
    return TRACKS[cfg.TRACK.TYPE](model)

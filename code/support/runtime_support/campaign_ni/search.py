from chavis.search.campaign_ni import PreparationIncomplete, ScientificFailure, SearchMethods as _SearchMethods
from chavis.search.policy import NORMAL_REACTION_MIC_IDPP


class SearchMethods(_SearchMethods):
    route_idpp_policy = NORMAL_REACTION_MIC_IDPP

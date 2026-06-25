import os


ATOMIC_SEEN_TASKS = frozenset({
    "CloseBlenderLid",
    "CloseFridge",
    "CloseToasterOvenDoor",
    "CoffeeSetupMug",
    "NavigateKitchen",
    "OpenCabinet",
    "OpenDrawer",
    "OpenStandMixerHead",
    "PickPlaceCounterToCabinet",
    "PickPlaceCounterToStove",
    "PickPlaceDrawerToCounter",
    "PickPlaceSinkToCounter",
    "PickPlaceToasterToCounter",
    "SlideDishwasherRack",
    "TurnOffStove",
    "TurnOnElectricKettle",
    "TurnOnMicrowave",
    "TurnOnSinkFaucet",
})


def is_atomic_seen_data_path(data_path: str) -> bool:
    parts = os.path.normpath(data_path).split(os.sep)
    return len(parts) >= 4 and parts[-4] == "atomic" and parts[-3] in ATOMIC_SEEN_TASKS

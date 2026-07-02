# UserView is an obj with an `id: str` field; provided elsewhere.
def compute_to_follow(allProfiles: list, followingIds: list, myProfileId: str) -> list:
    return [
        u
        for u in allProfiles
        if u.id not in followingIds and u.id != myProfileId
    ]

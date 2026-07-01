def display_username(profileUsername: str) -> str:
    return profileUsername.split("@")[0] if "@" in profileUsername else profileUsername

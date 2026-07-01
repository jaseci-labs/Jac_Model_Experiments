def tab_name(t: str) -> str:
    return ("Home" if t == "feed"
            else ("Explore" if t == "explore"
                  else ("Channels" if t == "channels" else "Profile")))

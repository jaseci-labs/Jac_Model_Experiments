"""RL driver (graph): social_graph.create_tweet.make.

Origin: this_is_jac/littlex/social_graph.jac. `create_tweet` navigates to the
caller's Profile and `make` attaches a new Tweet via a Post edge, grants write
permission, and reports its view. The marked body is `make`; the entry block
spawns it and prints the new tweet's deterministic fields. The new tweet's
created_at is a fixed literal and view ids are empty so the report is stable.
"""

obj TweetView {
    has id: str, content: str, author_username: str, created_at: str,
        likes: list[str], comments: list[dict[str, str]], is_mine: bool = False;
}

edge Post {}

node Profile {
    has username: str = "", bio: str = "", created_at: str = "";
}

node Tweet {
    has content: str = "", author_username: str = "", created_at: str = "",
        likes: list[str] = [], comments: list[dict[str, str]] = [];

    def to_view -> TweetView {
        mine = [root-->[?:Profile]];
        is_mine: bool = len(mine) > 0 and self.author_username == mine[0].username;
        return TweetView(
            id="", content=self.content, author_username=self.author_username,
            created_at=self.created_at, likes=self.likes, comments=self.comments, is_mine=is_mine
        );
    }
}

walker create_tweet {
    has content: str;

    can run with Root entry {
        visit [-->[?:Profile]];
    }

    can make with Profile entry {
        __HOLE__
    }
}

with entry {
    root ++> Profile(username="alice");
    res = root spawn create_tweet(content="hello jac");
    tv = res.reports[0];
    print("content:", tv.content);
    print("author:", tv.author_username);
    print("created_at:", tv.created_at);
    print("is_mine:", tv.is_mine);
    print("likes:", tv.likes);
}

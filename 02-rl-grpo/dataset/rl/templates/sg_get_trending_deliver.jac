"""RL driver (graph): social_graph.get_trending.deliver.

Origin: this_is_jac/littlex/social_graph.jac. `get_trending` tallies hashtags
across a profile's tweets and the `deliver` ability builds TrendingTag objects
from the counts, sorts them by descending count, and reports the top eight. The
marked body is `deliver`; the entry block builds a fixed graph and prints a
deterministic (tag, count) projection.
"""

obj TrendingTag { has tag: str, count: int; }
edge Post {}
node Profile { has username: str = ""; }
node Tweet { has content: str = ""; }

walker get_trending {
    has counts: dict[str, int] = {}, seen: dict[str, bool] = {},
        reports: list[list[TrendingTag]] = [];

    can run with Root entry {
        for r in allroots() {
            visit [r-->[?:Profile]-->[?:Tweet]];
        }
    }

    can tally with Tweet entry {
        tid = jid(here);
        if tid in self.seen {
            return;
        }
        self.seen[tid] = True;
        for word in here.content.split() {
            if word.startswith("#") and len(word) > 1 {
                tag = word.lower().rstrip(".,!?;:");
                self.counts[tag] = self.counts.get(tag, 0) + 1;
            }
        }
    }

    can deliver with Root exit {
        __HOLE__
    }
}

with entry {
    p = (root ++> Profile(username="alice"))[0];
    p ++> Tweet(content="loving #jac and #jac again, also #rl");
    p ++> Tweet(content="#rl scaling with #jac");
    p ++> Tweet(content="no tags here");
    res = root spawn get_trending();
    out = [(t.tag, t.count) for t in res.reports[0]];
    print(out);
}

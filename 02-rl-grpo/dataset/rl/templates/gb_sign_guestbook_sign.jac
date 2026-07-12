"""RL driver (guestbook): guestbook.sign_guestbook.sign.

Origin: this_is_jac/guestbook.jac. `sign_guestbook` walks into today's Day node
under root.shared (creating it on the day's first signature), then the `sign`
ability hangs a Visitor signature off that Day. The marked body is `sign`; the
entry block signs a fixed message and prints the deterministic guest entry.
"""

obj GuestEntry { has id: str, name: str, message: str, created_at: str; }

edge OnDay {}
edge Signed {}

node Day { has date: str = ""; }
node Visitor {
    has name: str = "", message: str = "", created_at: str = "";

    def to_view -> GuestEntry {
        return GuestEntry(id="", name=self.name, message=self.message, created_at=self.created_at);
    }
}

walker sign_guestbook {
    has name: str = "", message: str = "", reports: list[GuestEntry] = [];

    can run with Root entry {
        if not self.message.strip() {
            return;
        }
        today = "2020-01-01";
        visit [root-->[?:Day, date == today]] else {
            fresh_day = root +>: OnDay() :+> Day(date=today);
            grant(fresh_day[0], level=ConnectPerm);
            visit fresh_day;
        }
    }

    can sign with Day entry {
        __HOLE__
    }
}

with entry {
    res = root spawn sign_guestbook(name="  Alice  ", message="  hello world  ");
    ge = res.reports[0];
    print("name:", ge.name);
    print("message:", ge.message);
    print("created_at:", ge.created_at);

    anon = root spawn sign_guestbook(name="", message="second");
    print("anon_name:", anon.reports[0].name);

    day_count = len([root-->[?:Day]]);
    visitor_count = 0;
    for d in [root-->[?:Day]] {
        visitor_count += len([d-->[?:Visitor]]);
    }
    print("day_count:", day_count);
    print("visitor_count:", visitor_count);
}

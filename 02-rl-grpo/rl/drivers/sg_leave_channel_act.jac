"""RL driver (graph): social_graph.leave_channel.act.

Origin: this_is_jac/littlex/social_graph.jac. `leave_channel` subclasses
`find_channel` (resolves the target Channel by jid) and removes the caller's
Member edge on entry. The marked body is `act`; the entry block pre-joins a
fixed channel, leaves it, and prints the deterministic membership result.
"""

edge Member {}
node Profile { has username: str = ""; }
node Channel {
    has name: str = "", creator_username: str = "";
}

walker find_channel {
    has channel_id: str = "";

    can run with Root entry {
        if self.channel_id {
            target = jobj(self.channel_id);
            if isinstance(target, Channel) {
                visit [target];
            }
        }
    }
}

walker leave_channel(find_channel) {
    can act with Channel entry {
        # >>>HOLE id="sg_leave_channel_act" instruction="Implement the act ability (walker leave_channel, on Channel entry). Only act when jid(here) == self.channel_id. Find the caller's Profile via [root-->[?:Profile]][0]; collect the Member edges from that profile to here, and if any exist delete the first one. Then report {'success': True} and disengage."
        if jid(here) == self.channel_id {
            me = [root-->[?:Profile]][0];
            edges = [edge me->:Member:->here];
            if edges {
                del edges[0];
            }
            report {"success": True};
            disengage;
        }
        # <<<HOLE
    }
}

with entry {
    p = (root ++> Profile(username="alice"))[0];
    c = (root ++> Channel(name="jacdevs", creator_username="bob"))[0];
    p +>: Member() :+> c;
    cid = jid(c);
    print("before_member_count:", len([c<-:Member:<-[?:Profile]]));
    r1 = root spawn leave_channel(channel_id=cid);
    print("leave:", r1.reports[0]);
    print("after_member_count:", len([c<-:Member:<-[?:Profile]]));
    print("is_member:", len([edge p->:Member:->c]) > 0);
}

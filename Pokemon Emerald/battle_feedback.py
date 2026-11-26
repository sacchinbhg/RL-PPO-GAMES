"""Scalar combat feedback for the pixel-only Emerald environment."""
import struct


class BattleFeedback:
    def __init__(self):
        self.active = False
        self.unresolved = False
        self.paid = False
        self.previous = None
        self.wins = self.losses = self.damage_updates = 0
        self.total = 0.0

    def read(self, game, key):
        in_battle = key[0] == 6
        if in_battle and not self.active:
            self.active = True
            self.unresolved = self.paid = False
            self.previous = None
        if not self.active:
            return 0.0
        field = key[0] == 5 and game.verified_story(key) is not None
        if not in_battle and not field:
            self.previous = None
            return 0.0
        outcome = game.ram(0x0202433a, 1)[0]
        self.unresolved |= in_battle and outcome == 0
        feedback = 0.0
        if in_battle and self.unresolved and not self.paid:
            pair = []
            for address in (0x02024084, 0x020240dc):
                battler = game.ram(address, 0x2e)
                species, = struct.unpack_from("<H", battler)
                hp, = struct.unpack_from("<H", battler, 0x28)
                maximum, = struct.unpack_from("<H", battler, 0x2c)
                if not species or not 0 <= hp <= maximum <= 999 or not maximum:
                    pair = []
                    break
                pair.append((species, maximum, hp / maximum))
            if pair and self.previous and all(a[:2] == b[:2] for a,b in zip(pair,self.previous)):
                own_change = pair[0][2] - self.previous[0][2]
                opponent_change = pair[1][2] - self.previous[1][2]
                feedback = 0.5 * (own_change - opponent_change)
                self.damage_updates += int(feedback != 0)
            self.previous = pair
        if self.unresolved and outcome and not self.paid:
            self.wins += int(outcome == 1)
            self.losses += int(outcome == 2)
            feedback += int(outcome == 1) - int(outcome == 2)
            self.paid = True
        if field:
            self.active = False
            self.previous = None
        self.total += feedback
        return feedback

    def audit(self):
        return dict(won=self.wins, lost=self.losses, health_changes=self.damage_updates,
                    scalar_return=self.total)

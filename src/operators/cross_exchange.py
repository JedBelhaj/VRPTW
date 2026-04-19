
from operators.move_types import add_move
from utils.checker import evaluate_route


def generate_cross_exchange_moves(problem, routes, max_moves=120):
    moves = []

    for r1 in range(len(routes)):
        for r2 in range(r1 + 1, len(routes)):
            route1 = routes[r1]
            route2 = routes[r2]

            for len1 in (1, 2):
                for len2 in (1, 2):
                    if len(route1) <= len1 + 2 or len(route2) <= len2 + 2:
                        continue

                    for s1 in range(1, len(route1) - len1):
                        seg1 = route1[s1 : s1 + len1]
                        rem1 = route1[:s1] + route1[s1 + len1 :]

                        for s2 in range(1, len(route2) - len2):
                            seg2 = route2[s2 : s2 + len2]
                            rem2 = route2[:s2] + route2[s2 + len2 :]

                            cand1 = rem1[:s1] + seg2 + rem1[s1:]
                            cand2 = rem2[:s2] + seg1 + rem2[s2:]

                            f1, d1, _, _ = evaluate_route(problem, cand1)
                            if not f1:
                                continue
                            f2, d2, _, _ = evaluate_route(problem, cand2)
                            if not f2:
                                continue

                            stop = add_move(
                                moves,
                                routes,
                                [(r1, cand1), (r2, cand2)],
                                ("cross_exchange", tuple(seg1), tuple(seg2), r1, r2),
                                d1 + d2,
                                max_moves,
                            )
                            if stop:
                                return moves

    return moves


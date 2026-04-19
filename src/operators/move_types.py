from dataclasses import dataclass


@dataclass
class MoveCandidate:
    routes: list
    move_key: tuple
    objective: float


def add_move(moves, base_routes, updates, move_key, objective, max_moves):
    new_routes = [list(route) for route in base_routes]
    for idx, route in updates:
        new_routes[idx] = route

    moves.append(
        MoveCandidate(
            routes=new_routes,
            move_key=move_key,
            objective=objective,
        )
    )
    return len(moves) >= max_moves


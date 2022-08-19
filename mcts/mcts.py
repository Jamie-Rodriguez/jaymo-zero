from typing import TypedDict, List, TypeVar, Callable
from operator import itemgetter
from math import inf, sqrt, log
from functools import reduce, partial
from random import randint
from sys import maxsize
from tail_recursive import tail_recursive, FeatureSet


State = TypeVar("State")
Move = TypeVar("Move") # Also referred to as 'Action'

class Node(TypedDict):
    state: State
    num_rollouts: int
    score: int
    moves: List[dict] # list of Nodes (recursive)


'''
    Upper Confidence Bound 1 applied to trees,
    from Levente Kocsis and Csaba Szepesvári
'''
def uct(exploration, total_rollouts_parent, node):
    # type: (float, int, Node) -> float
    num_rollouts, score = itemgetter('num_rollouts', 'score')(node)

    if num_rollouts <= 0:
        return inf

    return score / num_rollouts + exploration * sqrt(log(total_rollouts_parent) / num_rollouts)

def pick_best_move(exploration, node):
    # type: (float, Node) -> int
    # TO-DO: We could just grab this from node['num_rollouts']...
    # is there a risk that node['num_rollouts'] isn't up to date?
    total_rollouts_parent = reduce(lambda sum, child: sum + child['num_rollouts'],
                                   node['moves'],
                                   0)
    get_uct = partial(uct, exploration, total_rollouts_parent)

    best_node_index = 0

    for i in range(len(node['moves'])):
        if get_uct(node['moves'][i]) > get_uct(node['moves'][best_node_index]):
            best_node_index = i

    return best_node_index

def pick_unexplored_move(get_random_int, get_valid_moves, is_terminal, node):
    # type: (Callable[[], int], Callable[[State], list[Move]], Callable[[State], bool], Node) -> Move
    if is_terminal(node['state']):
        return None

    valid_moves = get_valid_moves(node['state'])
    explored_moves = list(map(lambda n: n['move'], node['moves']))
    unexplored_moves = list(filter(lambda move: not move in explored_moves, valid_moves))

    if not unexplored_moves:
        return None

    return unexplored_moves[get_random_int() % len(unexplored_moves)]

def select(exploration, get_random_int, get_valid_moves, is_terminal, tree):
    # type: (float, Callable[[], int], Callable[[State], list[Move]], Callable[[State], bool], Node) -> list[Move]
    def loop(current_node, path): # type: (Node, list[Move]) -> list[Move]
        # If we have arrived at a not fully explored node or a terminal state
        if (len(get_valid_moves(current_node['state']))
           != len(current_node['moves'])) or is_terminal(current_node['state']):
            return path

        # Here we just grab current_node['num_rollouts'] instead of
        # calculating it from the child moves
        # If there is no possibility of current_node['num_rollouts'] being
        # out of date, then it is better to do this
        indexed_ucts = list(map(lambda index_node: {
            'index': index_node[0],
            'uct': uct(exploration, current_node['num_rollouts'], index_node[1])
        }, enumerate(current_node['moves'])))

        max_ucts = reduce(
            lambda max_values, current:
                max_values + [current] if current['uct'] == max_values[0]['uct']
                else [current] if current['uct'] > max_values[0]['uct']
                else max_values,
            indexed_ucts[1:],
            [indexed_ucts[0]])

        # There is a possibility that multiple moves may have the same statistics,
        # giving the same UCT values.
        # Settle the tie-break
        next_node = current_node['moves'][max_ucts[get_random_int() % len(max_ucts)]['index']]

        return loop(next_node, path + [next_node['move']])

    return loop(tree, [])

# Probably don't need unit tests for this tiny helper function...
def get_next_node(next_move, nodes):
    # type: (Move, list[Node]) -> Node | None
    def loop(nodes):
        # type: (list[Node]) -> Node | None
        if len(nodes) == 0:
            return None

        current_node = nodes[0]
        rest_nodes = nodes[1:]

        if current_node['move'] == next_move:
            return current_node

        if not rest_nodes:
            return None

        return loop(rest_nodes)

    return loop(nodes)

def treewalk(path, node):
    # type: (list[Move], Node) -> Node
    if not path:
        return node

    return treewalk(path[1:], get_next_node(path[0], node['moves']))

# This is the "expansion" step, but I think "replace node" is more indicative
# of *how* we are achieving it
# TO-DO: Protect against invalid path
def replace_node(node, path, updated_node): # type: (Node, list[Move], Node) -> Node
    if not path:
        return updated_node

    return {
        **node,
        # Note: It doesn't matter, but this conjoin operation changes the
        # original order of the moves
        'moves': list(filter(
                    lambda n: n['move'] != path[0],
                    node['moves'])) +
                 [replace_node(get_next_node(path[0], node['moves']),
                                             path[1:],
                                             updated_node)]
    }

def simulate (is_terminal,
              check_win,
              valid_moves,
              random_int,
              apply_move,
              initial_state):
    # type: (Callable[[State], bool], Callable[[State], int | None], Callable[[State], list[Move]], Callable[[], int], Callable[[State, Move], State], State) -> int | None

    def loop(state):
        # type: (State) -> int | None
        if is_terminal(state):
            return check_win(state)

        moves = valid_moves(state)
        next_move = moves[random_int() % len(moves)]

        return loop(apply_move(state, next_move))

    return loop(initial_state)

def is_path_valid(node, path):
    # type: (Node, list[Move]) -> bool
    if not path:
        return True

    next_node = get_next_node(path[0], node['moves'])

    if next_node is None:
        return False

    return is_path_valid(next_node, path[1:])

'''
  Note: The root node's score is not actually used, but we backprop up to it
  and update it anyway
  Use a depth-first search style recursion to drill down the tree along 'path',
  returning updated nodes as the call-stack collapses back up the tree to the
  root node
'''
# TO-DO: Change backprop to NOT update the score of the *root* node
# TO-DO: Protect against invalid path
'''
  If not all the moves on 'path' exist in the tree, will throw a null pointer
  exception when trying to access the non-existent current node, which = None
'''
def backprop(who_won, path, node, previous_player):
    # type: (int | None, list[Move], Node, int) -> Node
    new_node = {
        **node,
        'num_rollouts': node['num_rollouts'] + 1,
        'score': node['score'] + (1 if who_won == previous_player else
                                  0 if who_won is None else -1)
    }

    if not path:
        return new_node

    return {
        **new_node,
        # Note: It doesn't matter, but this conjoin operation changes the
        # original order of the moves
        'moves': list(filter(
                    lambda n: n['move'] != path[0],
                    new_node['moves'])) +
                # Get the next move along the path with updated stats
                [backprop(who_won,
                          path[1:],
                          get_next_node(path[0], new_node['moves']),
                          # Note: This is the *only* place in this MCTS code
                          # that expects State to have a specific property
                          # 'player_to_move'.
                          # This is a safe assumption for turn-based games'
                          # state
                          node['state']['player_to_move'])]
    }

def make_mcts_agent(exploration,
                    get_valid_moves,
                    is_terminal,
                    apply_move,
                    check_win,
                    player_index,
                    computation_budget):
    # type: (float, Callable[[State], list[Move]], Callable[[State], bool], Callable[[State, Move], State], Callable[[State], int | None], int, int) -> Callable[[State], Move]

    def get_random_int():
        # type: () -> int
        return randint(0, maxsize)

    def mcts(state):
        # type: (State) -> Move
        @tail_recursive(feature_set=FeatureSet.BASE)
        def loop(tree, budget):
            # type: (Node, int) -> Move
            if budget <= 0:
                # https://ai.stackexchange.com/questions/16905/mcts-how-to-choose-the-final-action-from-the-root
                # Choose best move via the "robust child" method = highest # of visits
                # Tie-break strategy: random choice
                max_rollouts = reduce(
                                lambda most_visited, current_node:
                                    most_visited + [current_node] if
                                        current_node['num_rollouts'] == most_visited[0]['num_rollouts']
                                    else [current_node] if
                                        current_node['num_rollouts'] > most_visited[0]['num_rollouts']
                                    else most_visited,
                                tree['moves'][1:],
                                [tree['moves'][0]])

                # There is a possibility that multiple moves may have the same statistics;
                # i.e. having the same number of rollouts.
                # Settle the tie-break
                next_node = max_rollouts[randint(0, len(max_rollouts) - 1)]

                return next_node['move']

            selected_node_path = select(exploration,
                                        get_random_int,
                                        get_valid_moves,
                                        is_terminal,
                                        tree)
            selected_node = treewalk(selected_node_path, tree)
            unexplored_move = pick_unexplored_move(get_random_int,
                                                   get_valid_moves,
                                                   is_terminal,
                                                   selected_node)
            # If selection picks a terminal state, unexplored move will be None.
            # Don't expand the selected node in this case (there is nothing to expand with!)
            path = selected_node_path + [unexplored_move] if unexplored_move else selected_node_path
            # Is there a way to do this functionally in Python??
            # Don't see any way without creating a function for this...
            new_tree = tree
            if unexplored_move:
                unexplored_node = { 'move': unexplored_move,
                                    'state': apply_move(selected_node['state'],
                                                        unexplored_move),
                                    'num_rollouts': 0,
                                    'score': 0,
                                    'moves': [] }
                expanded_node = {
                    **selected_node,
                    'moves': selected_node['moves'] + [unexplored_node]
                }

                new_tree = replace_node(tree, selected_node_path, expanded_node)
            # Simulate handles terminal nodes
            result = simulate(is_terminal,
                              check_win,
                              get_valid_moves,
                              get_random_int,
                              apply_move,
                              treewalk(path, new_tree)['state'])
            # The root node's score is not actually used, but we
            # backprop up to it and update it anyway.
            # We don't know the previous state, especially for the case
            # that the root node is the start of the game i.e. there
            # was not previous state
            return loop.tail_call(backprop(result, path, new_tree, -1),
                                  budget - 1)

        return loop({ 'state': state,
                      'num_rollouts': 0,
                      'score': 0,
                      'moves': [] },
                    computation_budget)

    return mcts

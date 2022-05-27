import numpy as np

import hardest_game


def play_game_AI(str, map_name='map1.txt'):
    game = hardest_game.Game(map_name=map_name, game_type='AI').run_AI_moves_graphic(moves=str)
    return game


def simulate(str, map_name='map1.txt'):
    game = hardest_game.Game(map_name=map_name, game_type='AI').run_AI_moves_no_graphic(moves=str)
    return game


def run_whole_generation(list_of_strs, N, map_name='map1.txt'):
    game = hardest_game.Game(map_name=map_name, game_type='AIS').run_generation(list_of_moves=list_of_strs, move_len=N)
    return game


def play_human_mode(map_name='map1.txt'):
    hardest_game.Game(map_name=map_name, game_type='player').run_player_mode()


def fitness(genome, goals_path, end_path, map_name="map1.txt"):
    temp = ""
    for i in genome:
        if i == 1:
            temp += 'w'
        elif i == 2:
            temp += 'a'
        elif i == 3:
            temp += 's'
        else:
            temp += 'd'

    game = simulate(temp, map_name=map_name)
    point = 0
    if game.hasDied:
        return 0
    if game.hasWon:
        return 10
    number = 0
    mini = np.inf
    for goal, reached in game.goals:
        if reached:
            point += 5
            number += 1
        else:
            temp = (game.player.x, game.player.y)
            dist = 0
            while temp != (-1, -1):
                temp = goals_path[(goal.x, goal.y)][temp]
                dist += 1
            if dist < mini:
                mini = dist
    if number == len(game.goals):
        temp = (game.player.x, game.player.y)
        dist = 0
        while temp != (-1, -1):
            temp = end_path[temp]
            dist += 1
        point += 1 / (dist // game.player.vel)
    else:
        point += 1 / (mini // game.player.vel)
    return point


def reverse_bfs_goals(goals, v_lines, h_lines, ply):
    accepted = list()
    for goal, is_reached in goals:
        if not is_reached:
            accepted.append(goal)
    result = dict()
    for goal in accepted:
        queue = []
        visited = []
        path = dict()
        s = (goal.x, goal.y)
        bfs(h_lines, path, ply, queue, s, v_lines, visited)
        result[(goal.x, goal.y)] = path
    return result


def bfs(h_lines, path, ply, queue, s, v_lines, visited):
    queue.append(s)
    visited.append(s)
    path[s] = (-1, -1)
    while queue:
        current = queue.pop(0)
        successors = [
            (current[0], current[1] + 1),
            (current[0], current[1] - 1),
            (current[0] + 1, current[1]),
            (current[0] - 1, current[1])
        ]
        for successor in successors:
            if successor[0] > 630:
                print("a")
            if successor in visited:
                continue
            move_type = ""
            if successor[0] > current[0]:
                move_type = 'd'
            elif successor[0] < current[0]:
                move_type = 'a'
            elif successor[1] > current[1]:
                move_type = 's'
            elif successor[1] < current[1]:
                move_type = 'w'
            is_on_board = on_borders(current, ply.width, ply.height, v_lines,
                                     h_lines, move_type, successor)
            if is_on_board:
                continue
            visited.append(successor)
            queue.append(successor)
            path[successor] = current


def special_bfs(h_lines, path, ply, queue, v, v_lines, visited):
    while queue:
        current = queue.pop(0)
        successors = [
            (current[0], current[1] + v),
            (current[0], current[1] - v),
            (current[0] + v, current[1]),
            (current[0] - v, current[1])
        ]
        for successor in successors:
            if successor in visited:
                continue
            move_type = ""
            if successor[0] > current[0]:
                move_type = 'd'
            elif successor[0] < current[0]:
                move_type = 'a'
            elif successor[1] > current[1]:
                move_type = 's'
            elif successor[1] < current[1]:
                move_type = 'w'
            is_on_board = on_borders(current, ply.width, ply.height, v_lines,
                                     h_lines, move_type, successor)
            if is_on_board:
                continue
            visited.append(successor)
            queue.append(successor)
            path[successor] = current


def reverse_bfs(end, v_lines, h_lines, ply):
    queue = []
    visited = []
    path = dict()
    s = (end.x, end.y)
    bfs(h_lines, path, ply, queue, s, v_lines, visited)
    return path


def on_borders(ply, width, height, vertical_lines, horizontal_lines, move_type, successor):
    l1 = vertical_lines
    l2 = horizontal_lines
    y = ply[1]
    x = ply[0]
    x2 = successor[0]
    y2 = successor[1]
    if move_type == 'a':
        t1 = False
        for l in l1:
            if y + height > l.y1 and y < l.y2 and x >= l.x1 > x2:
                return True
        if not t1:
            return False
    if move_type == 'd':
        t1 = False
        for l in l1:
            if y + height > l.y1 and y < l.y2 and x + width <= l.x1 < x2 + width:
                return True
        if not t1:
            return False
    if move_type == 's':
        t1 = False
        for l in l2:
            if x + width > l.x1 and x < l.x2 and y + height <= l.y1 < y2 + height:
                return True
        if not t1:
            return False
    if move_type == 'w':
        t1 = False
        for l in l2:
            if x + width > l.x1 and x < l.x2 and y >= l.y1 > y2:
                return True
        if not t1:
            return False


def mutate(genome, r_mut, ll):
    for i in range(ll, len(genome)):
        if np.random.rand() < r_mut:
            genome[i] = np.random.randint(1, 4, size=1)
    return genome


def crossover(p1, p2, r_cross, ll):
    c1, c2 = p1.copy(), p2.copy()
    if np.random.rand() < r_cross:
        pt = int(np.random.randint(ll, len(p1) - 2, 1))
        c1 = p1[:pt] + p2[pt:]
        c2 = p2[:pt] + p1[pt:]
    return c1, c2


def selection(pop, scores, k=3):
    selection_ix = int(np.random.randint(0, len(pop), 1))
    for ix in list(np.random.randint(0, len(pop), k - 1)):
        if scores[ix] < scores[selection_ix]:
            selection_ix = ix
    return pop[selection_ix]


def genetic_algorithm(objective, initial_length, n_pop, n_iter, r_cross, r_mut, inc_size, end_path, goals_path,
                      map_name):
    ll = 0
    pop = [np.random.randint(1, 4, size=initial_length).tolist() for _ in range(n_pop)]
    best, best_eval = 0, objective(pop[0], end_path=end_path, goals_path=goals_path, map_name=map_name)
    iteration = 0
    while True:
        scores = [objective(c, end_path=end_path, goals_path=goals_path, map_name=map_name) for c in pop]
        print(max(scores))
        for i in range(n_pop):
            if scores[i] < best_eval:
                best, best_eval = pop[i], scores[i]
                if best_eval == 10:
                    return best
        selected = [selection(pop, scores) for _ in range(n_pop)]
        children = list()
        for i in range(0, n_pop, 2):
            p1, p2 = selected[i], selected[i + 1]
            for c in crossover(p1, p2, r_cross, ll):
                mutate(c, r_mut, ll)
                children.append(c)
        pop = children
        if iteration == n_iter - 1:
            iteration = 0
            pop = increament_genomes(pop, inc_size)
            print(len(pop[0]))
            ll += inc_size
        iteration += 1


def increament_genomes(pop, inc_size):
    for genome in pop:
        genome.extend(np.random.randint(1, 4, inc_size))
    return pop


def optimize(map_name):
    game = simulate("", map_name)
    goals_res = reverse_bfs_goals(game.goals, game.Vlines, game.Hlines, game.player)
    end_res = reverse_bfs(game.end, game.Vlines, game.Hlines, game.player)
    return genetic_algorithm(fitness, initial_length=20,
                             n_pop=40,
                             n_iter=40,
                             inc_size=20,
                             r_cross=0.9,
                             r_mut=0.1,
                             map_name=map_name,
                             end_path=end_res,
                             goals_path=goals_res
                             )


def main():
    print(optimize("map1.txt"))
    print(optimize("map2.txt"))
    print(optimize("map3.txt"))


if __name__ == '__main__':
    main()

import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display

# Implemented methods
methods = ['DynProg', 'ValIter'];

# Some colours
LIGHT_RED    = '#FFC4CC';
LIGHT_GREEN  = '#95FD99';
BLACK        = '#000000';
WHITE        = '#FFFFFF';
LIGHT_PURPLE = '#E8D0FF';
LIGHT_ORANGE = '#FAE0C3';

class Maze:

    # Actions
    STAY       = 4 # error if stay = 0
    MOVE_LEFT  = 3
    MOVE_RIGHT = 2
    MOVE_UP    = 1
    MOVE_DOWN  = 0

    # Give names to actions
    actions_names = {
        STAY: "stay",
        MOVE_LEFT: "move left",
        MOVE_RIGHT: "move right",
        MOVE_UP: "move up",
        MOVE_DOWN: "move down"
    }

    # Reward values
    STEP_REWARD = -1
    GOAL_REWARD = 0
    IMPOSSIBLE_REWARD = -100
    LOST_REWARD = -200


    def __init__(self, maze, minotaur_stay = False, weights=None, random_rewards=False):
        """ Constructor of the environment Maze.
        """
        self.maze                     = maze;
        self.actions                  = self.__actions();
        self.minotaur_actions         = self.__minotaur_actions(minotaur_stay);
        self.states, self.map, self.minotaur_states, self.minotaur_map, self.state_map, self.states_complete = self.__states(); # added minotaur states and mapping
        self.n_actions                = len(self.actions);
        self.n_states                 = len(self.states);

        # add number of minotaur states and actions
        self.n_minotaur_states = len(self.minotaur_states);
        self.n_minotaur_actions = len(self.minotaur_actions);

        self.tot_states = len(self.states_complete);

        self.transition_probabilities = self.__transitions();
        self.rewards                  = self.__rewards(weights=weights,
                                                random_rewards=random_rewards);
        
        self.minotaur_rewards         = self.__minotaur_rewards();
    
        # added 2 new states    
        self.state_won = False; # not really necessary, the agent wins only if it reaches the exit within the time limit
        self.state_lost = False; # the agent loses when the minotaur cathces it or when it reaches the end of the time horizon without having found the exit
    

    def __actions(self):
        actions = dict();
        actions[self.STAY]       = (0, 0);
        actions[self.MOVE_LEFT]  = (0,-1);
        actions[self.MOVE_RIGHT] = (0, 1);
        actions[self.MOVE_UP]    = (-1,0);
        actions[self.MOVE_DOWN]  = (1,0);
        return actions;

    def __minotaur_actions(self, minotaur_stay):
        actions = dict()
        actions[self.MOVE_LEFT]  = (0,-1)
        actions[self.MOVE_RIGHT] = (0, 1)
        actions[self.MOVE_UP]    = (-1,0)
        actions[self.MOVE_DOWN]  = (1,0)
        if minotaur_stay:
            actions[self.STAY]   = (0, 0)
        return actions
    
    def __minotaur_move(self, state, action):
        # 1) check number of allowed moves (i.e. it cannot go outside the maze but it can go through walls)
        # 2) return selected state

        row = self.minotaur_states[state][0] + self.minotaur_actions[action][0];
        col = self.minotaur_states[state][1] + self.minotaur_actions[action][1];
        # Is the future position an impossible one ?
        hitting_maze_walls =  (row == -1) or (row == self.maze.shape[0]) or \
                                (col == -1) or (col == self.maze.shape[1])
        if not hitting_maze_walls:
            return self.minotaur_map[(row, col)], (row, col)
        else:
            return state, self.minotaur_states[state];

    
    def __minotaur_random_move(self, state):
        moves = self.__minotaur_possible_actions(state);
            
        # uniformly select any element from the list 
        next_move = np.random.choice(moves) 
        row = self.minotaur_states[state][0] + self.minotaur_actions[next_move][0];
        col = self.minotaur_states[state][1] + self.minotaur_actions[next_move][1];

        #print("R-C: ",row,"-", col)

        return self.minotaur_map[(row, col)], (row, col)


    def __states(self):
        states = dict();
        map = dict();

        # add the position of the minotaur to the state of the agent
        minotaur_states = dict();
        minotaur_map = dict();

        #mapping of both agent and minotaur states
        state_map = dict();
        states_complete = dict();

        end = False;
        s = 0;
        s_m = 0;
        s_map = 0;
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
            
                minotaur_states[s_m] = (i,j); # the minotaur can move in any cell
                minotaur_map[(i,j)] = s_m;
                s_m+=1;
                
                # not obstacle 
                if self.maze[i,j] != 1:
                    for k in range(self.maze.shape[0]):
                        for l in range(self.maze.shape[1]):
                            if (((i!=k) or (j!=l)) and (self.maze[i,j] != 2)): #not lost and not won
                                states_complete[s_map] = (i,j),(k,l);
                                state_map[(i,j),(k,l)] = s_map;
                                s_map+=1;
                
                    states[s] = (i,j);
                    map[(i,j)] = s;
                    s += 1;
        
                state_map[(i,j)] = s_m + s;
        #add lost state
        state_map['lost'] = s_map;
        states_complete[s_map] = 'lost';

        # add won state
        s_map+=1;
        state_map['won'] = s_map;
        states_complete[s_map] = 'won';

        print("States ", states_complete);
        return states, map, minotaur_states, minotaur_map, state_map, states_complete

    def __move(self, state, action):
        """ Makes a step in the maze, given a current position and an action.
            If the action STAY or an inadmissible action is used, the agent stays in place.

            :return tuple next_cell: Position (x,y) on the maze that agent transitions to.
        """
        # Compute the future position given current (state, action)
        row = self.states[state][0] + self.actions[action][0];
        col = self.states[state][1] + self.actions[action][1];
        # Is the future position an impossible one ?
        hitting_maze_walls =  (row == -1) or (row == self.maze.shape[0]) or \
                              (col == -1) or (col == self.maze.shape[1]) or \
                              (self.maze[row,col] == 1);
        # Based on the impossiblity check return the next state.
        if hitting_maze_walls:
            return state, self.states[state];
        else:
            return self.map[(row, col)], (row,col);

    def __minotaur_possible_actions(self, state):  
        moves = list()
        # 1) check number of allowed moves (i.e. it cannot go outside the maze but it can go through walls)
            # try all possible moves
        for el in self.minotaur_actions:
            # check for every move if its allowed
            # add to a list only the allowed next states
        # 2) select a random move between the available ones
            # random element from a list (uniform random)
        # 3) return selected state

            row = self.minotaur_states[state][0] + self.minotaur_actions[el][0];
            col = self.minotaur_states[state][1] + self.minotaur_actions[el][1];
            # Is the future position an impossible one ?
            hitting_maze_walls =  (row == -1) or (row == self.maze.shape[0]) or \
                                  (col == -1) or (col == self.maze.shape[1])
            if not hitting_maze_walls:
                moves.append(el)

        return moves

    def __transitions(self):
        """ Computes the transition probabilities for every state action pair.
            :return numpy.tensor transition probabilities: tensor of transition
            probabilities of dimension S*S*A
        """
        # Initialize the transition probailities tensor (S,S,A)
        dimensions = (self.tot_states, self.tot_states, self.n_actions);
        transition_probabilities = np.zeros(dimensions);

        # Compute the transition probabilities. Note that the transitions are deterministic
        for state in range(self.tot_states):
            if((self.states_complete[state] != 'lost') and (self.states_complete[state] != 'won')):
                #actual state
                (x,y),(x_m,y_m) = self.states_complete[state];
                
                
                for a in range(self.n_actions):
                    _, (x_next,y_next) = self.__move(self.map[(x, y)],a);

                    possible_actions = self.__minotaur_possible_actions(self.minotaur_map[(x_m,y_m)]);
                    for minotaur_action in possible_actions:

                        _, (x_m_next, y_m_next) = self.__minotaur_move(self.minotaur_map[(x_m,y_m)], minotaur_action);

                        #lost
                        # p(lost|state\{lost},a) = theta if ((x == x_m + 1) or (x == x_m-1) or (y == y_m+1) or (y == y_m-1))
                        # 0 otherwise
                        #theta = 1/minotaur possible actions
                        if(x_next==x_m_next and y_next == y_m_next):
                            next_state = self.state_map['lost'];
                            transition_probabilities[next_state, state, a] = 1/len(possible_actions);
                        else:
                            # won?
                            if(self.maze[x_next,y_next] ==2):
                                next_state = self.state_map['won'];
                            else:
                                next_state = self.state_map[(x_next,y_next),(x_m_next, y_m_next)];

                            if ((x == x_m + 1) or (x == x_m-1) or (y == y_m+1) or (y == y_m-1)):
                                transition_probabilities[next_state, state, a] = 1 - 1/len(possible_actions);
                            else: 
                                transition_probabilities[self.state_map['lost'], state, a] = 0;
                                transition_probabilities[next_state, state, a] = 1;                     

            else: 
                # p(lost|lost,a) = 1 for any action
                # p(won|won,a) = 1 for any action
                for a in range(self.n_actions):
                    transition_probabilities[state, state, a] = 1;                        
        
        return transition_probabilities;


    def __minotaur_rewards(self):

        rewards = np.zeros((self.tot_states, self.n_actions));

        # If the rewards are not described by a weight matrix
        for s in range(self.tot_states):
            for a in range(self.n_actions):
                if(self.states_complete[s] != 'lost'):
                    if(self.states_complete[s] == 'won'): # the reward for winning for any possible action
                        rewards[s,a] = self.GOAL_REWARD;
                    
                    else:
                        #actual state
                        (x,y), _ = self.states_complete[s]; 
                        next_s, _ = self.__move(self.map[(x, y)],a); #agent next position
                    
                        # Rewrd for hitting a wall
                        if s == next_s and a != self.STAY:
                            rewards[s,a] = self.IMPOSSIBLE_REWARD;
                        else:
                            rewards[s,a] = self.STEP_REWARD;
                else:
                    # reward of losing for any action 
                    rewards[self.state_map['lost'],a] = self.LOST_REWARD;

        return rewards;

    def __rewards(self, weights=None, random_rewards=None):

        rewards = np.zeros((self.n_states, self.n_actions));

        # If the rewards are not described by a weight matrix
        if weights is None:
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    next_s = self.__move(s,a);
                    # Rewrd for hitting a wall
                    if s == next_s and a != self.STAY:
                        rewards[s,a] = self.IMPOSSIBLE_REWARD;
                    # Reward for reaching the exit
                    elif s == next_s and self.maze[self.states[next_s]] == 2:
                        rewards[s,a] = self.GOAL_REWARD;
                    # Reward for taking a step to an empty cell that is not the exit
                    else:
                        rewards[s,a] = self.STEP_REWARD;

                    # If there exists trapped cells with probability 0.5
                    if random_rewards and self.maze[self.states[next_s]]<0:
                        row, col = self.states[next_s];
                        # With probability 0.5 the reward is
                        r1 = (1 + abs(self.maze[row, col])) * rewards[s,a];
                        # With probability 0.5 the reward is
                        r2 = rewards[s,a];
                        # The average reward
                        rewards[s,a] = 0.5*r1 + 0.5*r2;
        # If the weights are descrobed by a weight matrix
        else:
            for s in range(self.n_states):
                 for a in range(self.n_actions):
                     next_s = self.__move(s,a);
                     i,j = self.states[next_s];
                     # Simply put the reward as the weights o the next state.
                     rewards[s,a] = weights[i][j];

        return rewards;

    # added minotaur
    def simulate(self, start, policy, method, minotaur_start = False):
        if method not in methods:
            error = 'ERROR: the argument method must be in {}'.format(methods);
            raise NameError(error);

        path = list();
        minotaur_path = list();

        if method == 'DynProg':
            # Deduce the horizon from the policy shape
            horizon = policy.shape[1];
            # Initialize current state and time
            t = 0;
            s = self.map[start]; #tot map
            # Add the starting position in the maze to the path
            path.append(start);

            # Add staring position of the minotaur to the minotaur path
            if minotaur_start:
                minotaur_path.append(minotaur_start);
                minotaur_state = minotaur_start;

            while t < horizon-1:
                # Move to next state given the policy and the current state
                next_s, _ = self.__move(s,policy[s,t]);
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append(self.states[next_s]);

                #Minotaur random move
                if minotaur_start:
                    next_minotaur,_ = self.__minotaur_random_move(minotaur_state);
                    minotaur_path.append(next_minotaur);

                    minotaur_state = next_minotaur;

                # Update time and state for next iteration
                t +=1;
                s = next_s;
        if method == 'ValIter':
            # Initialize current state, next state and time
            t = 1;
            s = self.map[start];
            # Add the starting position in the maze to the path
            path.append(start);
            # Move to next state given the policy and the current state
            next_s = self.__move(s,policy[s]);
            # Add the position in the maze corresponding to the next state
            # to the path
            path.append(self.states[next_s]);
            # Loop while state is not the goal state
            while s != next_s:
                # Update state
                s = next_s;
                # Move to next state given the policy and the current state
                next_s = self.__move(s,policy[s]);
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append(self.states[next_s])
                # Update time and state for next iteration
                t +=1;
        return path, minotaur_path

    def simulate_minotaur(self, start, minotaur_start, policy, method):
        lost_won = False;

        if method not in methods:
            error = 'ERROR: the argument method must be in {}'.format(methods);
            raise NameError(error);

        path = list();

        minotaur_path = list();

        if method == 'DynProg':
            # Deduce the horizon from the policy shape
            horizon = policy.shape[1];
            # Initialize current state and time
            t = 0;
            s = self.state_map[start, minotaur_start]; 
            state = self.map[start];
            # Add the starting position in the maze to the path
            path.append(start);

            # Add staring position of the minotaur to the minotaur path
            minotaur_path.append(minotaur_start);
            minotaur_state = self.minotaur_map[minotaur_start];

            while t < horizon-1 and not lost_won:
                # Move to next state given the policy and the current state
                state, (x_next,y_next) = self.__move(state,policy[s,t]);
                # Add the position in the maze corresponding to the next state to the path
                path.append((x_next,y_next));

                # Minotaur random move
                minotaur_state, (x_m,y_m) = self.__minotaur_random_move(minotaur_state);
                minotaur_path.append((x_m,y_m));
                

                # Update time and state for next iteration
                t +=1;
                # lost
                if (x_next == x_m and y_next == y_m):
                    s = self.state_map['lost'];
                    print('LOST')
                    lost_won = True;
                else:
                    if (self.maze[x_next,y_next] == 2): #won
                        s = self.state_map['won'];
                        print('WON');
                        lost_won = True;
                    else:
                        s = self.state_map[(x_next,y_next), (x_m,y_m)];
        return path, minotaur_path

    def show(self):
        print('The states are :')
        print(self.states)
        print('The actions are:')
        print(self.actions)
        print('The mapping of the states:')
        print(self.map)
        print('The rewards:')
        print(self.rewards)

def dynamic_programming(env, horizon):
    """ Solves the shortest path problem using dynamic programming
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input int horizon        : The time T up to which we solve the problem.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """

    # The dynamic prgramming requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p         = env.transition_probabilities;
    r         = env.rewards;
    n_states  = env.n_states;
    n_actions = env.n_actions;
    T         = horizon;

    # The variables involved in the dynamic programming backwards recursions
    V      = np.zeros((n_states, T+1));
    policy = np.zeros((n_states, T+1));
    Q      = np.zeros((n_states, n_actions));


    # Initialization
    Q            = np.copy(r);
    V[:, T]      = np.max(Q,1);
    policy[:, T] = np.argmax(Q,1);

    # The dynamic programming bakwards recursion
    for t in range(T-1,-1,-1):
        # Update the value function acccording to the bellman equation
        for s in range(n_states):
            for a in range(n_actions):
                # Update of the temporary Q values
                Q[s,a] = r[s,a] + np.dot(p[:,s,a],V[:,t+1])
        # Update by taking the maximum Q value w.r.t the action a
        V[:,t] = np.max(Q,1);
        # The optimal action is the one that maximizes the Q function
        policy[:,t] = np.argmax(Q,1);
    return V, policy;

def dynamic_programming_minotaur(env, horizon):
    """ Solves the shortest path problem using dynamic programming
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input int horizon        : The time T up to which we solve the problem.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """

    # The dynamic prgramming requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p         = env.transition_probabilities;
    r         = env.minotaur_rewards;
    n_states  = env.tot_states;
    n_actions = env.n_actions;
    T         = horizon;

    # The variables involved in the dynamic programming backwards recursions
    V      = np.zeros((n_states, T+1));
    policy = np.zeros((n_states, T+1));
    Q      = np.zeros((n_states, n_actions));


    # Initialization
    Q            = np.copy(r);
    V[:, T]      = np.max(Q,1);
    policy[:, T] = np.argmax(Q,1);

    print("Rewards shape [s,a]",r.shape)
    print("Transition prob shape [s,s,a]", p.shape)
    print("V:",V[:,T])
    print("policy at T, action that max the rewards at the end", policy[:,T])

    # The dynamic programming bakwards recursion
    for t in range(T-1,-1,-1):
        # Update the value function acccording to the bellman equation
        for s in range(n_states):
            for a in range(n_actions):
                # Update of the temporary Q values
                Q[s,a] = r[s,a] + np.dot(p[:,s,a],V[:,t+1])
        # Update by taking the maximum Q value w.r.t the action a
        V[:,t] = np.max(Q,1);
        # The optimal action is the one that maximizes the Q function
        policy[:,t] = np.argmax(Q,1);
    return V, policy;

def draw_policy(env, minotaur_position, policy, time):
    maze = env.maze;
    (x_m,y_m) = minotaur_position;
    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED};

    # Give a color to each cell
    rows,cols    = maze.shape;
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows));

    # Remove the axis ticks and add title title
    ax = plt.gca();
    ax.set_title('The Maze');
    ax.set_xticks([]);
    ax.set_yticks([]);

    # Give a color to each cell
    rows,cols    = maze.shape;
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                            cellColours=colored_maze,
                            cellLoc='center',
                            loc=(0,0),
                            edges='closed');
    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows);
        cell.set_width(1.0/cols);
    
    # policy in every state in which the minotaur is in that position
    for el in range(env.n_states):
        (x,y) = env.states[el];
        if((x!=x_m) or (y!=y_m)):
            if(env.maze[x,y] != 2): # not in won cell
                state = env.state_map[(x,y),(x_m,y_m)];
                action = policy[state,time];
                if action == env.MOVE_DOWN:
                    grid.get_celld()[(x,y)].get_text().set_text('\u2193');
                if action == env.MOVE_UP:
                    grid.get_celld()[(x,y)].get_text().set_text('\u2191');
                if action == env.MOVE_RIGHT:
                    grid.get_celld()[(x,y)].get_text().set_text('\u2192');
                if action == env.MOVE_LEFT:
                    grid.get_celld()[(x,y)].get_text().set_text('\u2190');
                if action == env.STAY:
                    grid.get_celld()[(x,y)].get_text().set_text('-');
            else:
                grid.get_celld()[(x,y)].set_facecolor(LIGHT_GREEN)
                grid.get_celld()[(x,y)].get_text().set_text('Won')

    grid.get_celld()[(x_m,y_m)].set_facecolor(LIGHT_PURPLE)
    grid.get_celld()[(x_m,y_m)].get_text().set_text('Minotaur')
    
    
    return 0


def value_iteration(env, gamma, epsilon):
    """ Solves the shortest path problem using value iteration
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input float gamma        : The discount factor.
        :input float epsilon      : accuracy of the value iteration procedure.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """
    # The value itearation algorithm requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p         = env.transition_probabilities;
    r         = env.rewards;
    n_states  = env.n_states;
    n_actions = env.n_actions;

    # Required variables and temporary ones for the VI to run
    V   = np.zeros(n_states);
    Q   = np.zeros((n_states, n_actions));
    BV  = np.zeros(n_states);
    # Iteration counter
    n   = 0;
    # Tolerance error
    tol = (1 - gamma)* epsilon/gamma;

    # Initialization of the VI
    for s in range(n_states):
        for a in range(n_actions):
            Q[s, a] = r[s, a] + gamma*np.dot(p[:,s,a],V);
    BV = np.max(Q, 1);

    # Iterate until convergence
    while np.linalg.norm(V - BV) >= tol and n < 200:
        # Increment by one the numbers of iteration
        n += 1;
        # Update the value function
        V = np.copy(BV);
        # Compute the new BV
        for s in range(n_states):
            for a in range(n_actions):
                Q[s, a] = r[s, a] + gamma*np.dot(p[:,s,a],V);
        BV = np.max(Q, 1);
        # Show error
        #print(np.linalg.norm(V - BV))

    # Compute policy
    policy = np.argmax(Q,1);
    # Return the obtained policy
    return V, policy;

def draw_maze(maze):

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED};

    # Give a color to each cell
    rows,cols    = maze.shape;
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows));

    # Remove the axis ticks and add title title
    ax = plt.gca();
    ax.set_title('The Maze');
    ax.set_xticks([]);
    ax.set_yticks([]);

    # Give a color to each cell
    rows,cols    = maze.shape;
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                            cellColours=colored_maze,
                            cellLoc='center',
                            loc=(0,0),
                            edges='closed');
    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows);
        cell.set_width(1.0/cols);

def animate_solution(maze, path, minotaur_path = False):
    print(path)
    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED};

    # Size of the maze
    rows,cols = maze.shape;

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows));

    # Remove the axis ticks and add title title
    ax = plt.gca();
    ax.set_title('Policy simulation');
    ax.set_xticks([]);
    ax.set_yticks([]);

    # Give a color to each cell
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                     cellColours=colored_maze,
                     cellLoc='center',
                     loc=(0,0),
                     edges='closed');

    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows);
        cell.set_width(1.0/cols);


    # Update the color at each frame
    for i in range(len(path)):
        grid.get_celld()[(path[i])].set_facecolor(LIGHT_ORANGE)
        grid.get_celld()[(path[i])].get_text().set_text('Player')

    # Minotaur
        if minotaur_path:
            grid.get_celld()[(minotaur_path[i])].set_facecolor(LIGHT_PURPLE)
            grid.get_celld()[(minotaur_path[i])].get_text().set_text('Minotaur')

        if i > 0:
            if maze[path[i]] == 2:
                grid.get_celld()[(path[i])].set_facecolor(LIGHT_GREEN)
                grid.get_celld()[(path[i])].get_text().set_text('Player is out')
            
            if path[i-1] != path[i]:
                #clear path
                grid.get_celld()[(path[i-1])].set_facecolor(col_map[maze[path[i-1]]])
                grid.get_celld()[(path[i-1])].get_text().set_text('')
            # clear minotaur path
            if minotaur_path and minotaur_path[i-1] != minotaur_path[i]:
                grid.get_celld()[(minotaur_path[i-1])].set_facecolor(col_map[maze[minotaur_path[i-1]]])
                grid.get_celld()[(minotaur_path[i-1])].get_text().set_text('')
            
                if minotaur_path[i] == path[i]:
                    grid.get_celld()[(path[i])].set_facecolor(LIGHT_RED)
                    grid.get_celld()[(path[i])].get_text().set_text('Lost')

        display.display(fig)
        display.clear_output(wait=True)
        time.sleep(1)

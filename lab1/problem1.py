# KTH Royal Institute of Technology
# EL2805 - Reinforcement Learning
# Period 2 - 2023
# Lab 1, ex 1
# Valeria Grotto 200101266021
# Dalim Whaby 19970606-T919


import numpy as np
import matplotlib.pyplot as plt
import time
import random
import math
from IPython import display

# Implemented methods
methods = ['DynProg', 'ValIter', 'Qlearning', 'sarsa'];

# Some colours
LIGHT_RED    = '#FFC4CC';
LIGHT_GREEN  = '#95FD99';
BLACK        = '#000000';
WHITE        = '#FFFFFF';
LIGHT_PURPLE = '#E8D0FF';
LIGHT_ORANGE = '#FAE0C3';
YELLOW       = '#FDFD96';

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
    IMPOSSIBLE_REWARD = -100000
    LOST_REWARD = -1000
    KEY_REWARD = 500


    def __init__(self, maze, key = False, minotaur_stay = False, weights=None, random_rewards=False):
        """ Constructor of the environment Maze.
        """
        self.maze                     = maze;
        self.actions                  = self.__actions();
        self.minotaur_actions         = self.__minotaur_actions(minotaur_stay);

        if key:
            self.states, self.map, self.minotaur_states, self.minotaur_map, self.state_map, self.states_complete = self.__states_key(); # added minotaur states and mapping
        else: 
            self.states, self.map, self.minotaur_states, self.minotaur_map, self.state_map, self.states_complete = self.__states(); # added minotaur states and mapping
        self.n_actions                = len(self.actions);
        self.n_states                 = len(self.states);

        # add number of minotaur states and actions
        self.n_minotaur_states = len(self.minotaur_states);
        self.n_minotaur_actions = len(self.minotaur_actions);

        self.tot_states = len(self.states_complete);

        self.transition_probabilities = self.__transitions(key);
        self.minotaur_rewards         = self.__minotaur_rewards(key);
    
    
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

    
    def __minotaur_random_move(self, state, key = False):
        if key:
            target_state, state_minotaur , _=  self.states_complete[state]
        else:
            target_state, state_minotaur=  self.states_complete[state]

        moves, best_moves = self.__minotaur_possible_actions(self.minotaur_map[state_minotaur], target_state);

        # if we have the key the minotaur moves randomly with prob 0.65 otherwise moves towards you
        if key:
            # generate random number
            n = random.uniform(0, 1);
            # minotaur possible actions

            if n < 0.65: # select random action from the action space
                # uniformly select any element from the list 
                next_move = np.random.choice(moves) 
            else: # select best action
                # uniformly select any element from the list 
                if len(best_moves)>0:
                    next_move = np.random.choice(best_moves) 
                else: 
                    next_move = np.random.choice(moves) 
        else:
            # uniformly select any element from the list 
            next_move = np.random.choice(moves) 
        
        row = state_minotaur[0] + self.minotaur_actions[next_move][0];
        col = state_minotaur[1] + self.minotaur_actions[next_move][1];

        # print("Minotaur move: ",row,"-", col, "previous state:",  state_minotaur, "action:", next_move)

        return self.minotaur_map[(row, col)], (row, col)
    
    def __states_key(self):
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
                            for key in range(2):
                                if (((i!=k) or (j!=l)) and not (self.maze[i,j] == 2 and key)): #not lost and not won
                                    states_complete[s_map] = (i,j),(k,l),key;
                                    state_map[(i,j),(k,l),key] = s_map;
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

        # print("States ", states_complete);
        return states, map, minotaur_states, minotaur_map, state_map, states_complete


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

        # print("States ", states_complete);
        return states, map, minotaur_states, minotaur_map, state_map, states_complete

    def __move(self, state, action):
        """ Makes a step in the maze, given a current position and an action.
            If the action STAY or an inadmissible action is used, the agent stays in place.

            :return tuple next_cell: Position (x,y) on the maze that agent transitions to.
        """
        # Compute the future position given current (state, action)
        row = self.states[state][0] + self.actions[action][0];
        col = self.states[state][1] + self.actions[action][1];
        # Is the future position not possible (hit a wall) ?
        hitting_maze_walls =  (row == -1) or (row == self.maze.shape[0]) or \
                              (col == -1) or (col == self.maze.shape[1]) or \
                              (self.maze[row,col] == 1);

        # Based on the impossiblity check return the next state.
        if hitting_maze_walls:
            return state, self.states[state];
        else:
            return self.map[(row, col)], (row,col);

    def __minotaur_possible_actions(self, state, target_state):  
        '''
            returns a list of possible moves and the move that brings the minotaur close to the agent/target
        '''
        moves = list()
        best_moves = list()
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

            distance_before_move = math.dist(self.minotaur_states[state], target_state)
            distance_after_move = math.dist((row, col), target_state)

            # Is the future position an impossible one ?
            hitting_maze_walls =  (row == -1) or (row == self.maze.shape[0]) or \
                                  (col == -1) or (col == self.maze.shape[1])
            if not hitting_maze_walls:
                moves.append(el)
                # Calculate Euclidean distance

                if distance_after_move < distance_before_move:
                    best_moves.append(el)

        # print("state:", self.minotaur_states[state]," -",  target_state, "=> best actions:", best_moves)
 
        return moves, best_moves


    def __transitions(self,key=False):
        """ Computes the transition probabilities for every state action pair.
            :return numpy.tensor transition probabilities: tensor of transition
            probabilities of dimension S*S*A
        """

        # the probability of going to a next state given an action is not 1, we have to consider the minotaur random move

        # Initialize the transition probailities tensor (S,S,A)
        dimensions = (self.tot_states, self.tot_states, self.n_actions);
        transition_probabilities = np.zeros(dimensions);

        # Compute the transition probabilities. Note that the transitions are deterministic
        for state in range(self.tot_states):
            if((self.states_complete[state] != 'lost') and (self.states_complete[state] != 'won')):
                #actual state
                if key:
                    (x,y),(x_m,y_m), key_val = self.states_complete[state];
                else:
                    (x,y),(x_m,y_m) = self.states_complete[state];
                
                
                for a in range(self.n_actions):
                    _, (x_next,y_next) = self.__move(self.map[(x, y)],a);

                    #best action is the one that shortens the distance from the agent
                    possible_actions, best_actions = self.__minotaur_possible_actions(self.minotaur_map[(x_m,y_m)], (x,y));
                    for minotaur_action in possible_actions:

                        _, (x_m_next, y_m_next) = self.__minotaur_move(self.minotaur_map[(x_m,y_m)], minotaur_action);

                        if key:
                            if len(best_actions)> 0 and (minotaur_action in best_actions):
                                p_state = 0.35 * (1/len(best_actions)) + 0.65*(1/len(possible_actions))
                            else:
                                p_state = 0.65*(1/len(possible_actions))
                        else:
                            p_state = 1/len(possible_actions)

                        #lost
                        if(x_next==x_m_next and y_next == y_m_next):
                            next_state = self.state_map['lost'];
                            transition_probabilities[next_state, state, a] = p_state;
                        else:
                            if key:
                                # pick the key
                                if self.maze[x_next,y_next] == 3 and key_val == 0: 
                                    next_state = self.state_map[(x_next,y_next),(x_m_next, y_m_next),1];
                                    transition_probabilities[next_state, state, a] = p_state;
                                # winning cell
                                elif self.maze[x_next,y_next] == 2 and key_val: 
                                    next_state = self.state_map['won'];
                                    # otherwise is a normal cell
                                    if ((x == x_m + 1) or (x == x_m-1) or (y == y_m+1) or (y == y_m-1)):
                                        transition_probabilities[next_state, state, a] = 1-p_state;
                                    else:
                                        transition_probabilities[next_state, state, a] = 1;
                                else:
                                    next_state = self.state_map[(x_next,y_next),(x_m_next, y_m_next),key_val];
                                    transition_probabilities[next_state, state, a] = p_state;
                            else:
                                # won
                                if self.maze[x_next,y_next] == 2:
                                    next_state = self.state_map['won'];
                                    if ((x == x_m + 1) or (x == x_m-1) or (y == y_m+1) or (y == y_m-1)):
                                        transition_probabilities[self.state_map['won'], state, a] = 1-p_state;
                                    else:
                                        transition_probabilities[self.state_map['won'], state, a] = 1;
                                else:
                                    next_state = self.state_map[(x_next,y_next),(x_m_next, y_m_next)];
                                    transition_probabilities[next_state, state, a] = p_state; # prob of going to the next state

            else: 
                # p(lost|lost,a) = 1 for any action
                # p(won|won,a) = 1 for any action
                for a in range(self.n_actions):
                    transition_probabilities[state, state, a] = 1;                        
        
        return transition_probabilities;


    def __minotaur_rewards(self,key):

        rewards = np.zeros((self.tot_states, self.n_actions));

        # If the rewards are not described by a weight matrix
        for s in range(self.tot_states):
            for a in range(self.n_actions):
                if(self.states_complete[s] == 'lost'):
                    rewards[self.state_map['lost'],a] = self.LOST_REWARD;
                elif (self.states_complete[s] == 'won'):
                    rewards[s,a] = self.GOAL_REWARD;
                else:
                    #actual state
                    if key:
                        (x,y), _, key_val = self.states_complete[s]; 
                    else:
                        (x,y), _ = self.states_complete[s]; 
                    state = self.map[(x, y)];
                    next_s, _ = self.__move(state,a); #agent next position
                
                    # Rewrd for hitting a wall
                    if state == next_s and a != self.STAY:
                        rewards[s,a] = self.IMPOSSIBLE_REWARD;
                    else:
                        #key
                        if key and self.maze[self.states[next_s]] == 3 and key_val == 0:
                            rewards[s,a] = self.KEY_REWARD;
                        else: #step
                            rewards[s,a] = self.STEP_REWARD;

        return rewards;

    def simulate_minotaur(self, start, minotaur_start, policy, method, key = False, key_cell = (0,6)):
        if(key):
            self.maze[key_cell] = 3;
        lost_won = False;
        lost = False;
        won = False;

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
            
            if key:
                key_val = 0;
                s = self.state_map[start, minotaur_start, key_val]; 
            else:
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
                minotaur_state, (x_m,y_m) = self.__minotaur_random_move(s, key);
                minotaur_path.append((x_m,y_m));
                

                # Update time and state for next iteration
                t +=1;
                # lost
                if (x_next == x_m and y_next == y_m):
                    s = self.state_map['lost'];
                    lost_won = True;
                    lost = True;
                else:
                    if key:
                        if (self.maze[x_next,y_next] == 2 and key_val): # won with key
                            s = self.state_map['won'];
                            lost_won = True;
                            won = True;
                        else:
                            if self.maze[(x_next,y_next)]==3:
                                key_val = 1;
                                # self.maze[(x_next,y_next)]=0; # since we got the key the cell is resetted
                            s = self.state_map[(x_next,y_next), (x_m,y_m), key_val];
                    
                    else: 
                        if (self.maze[x_next,y_next] == 2): #won
                            s = self.state_map['won'];
                            lost_won = True;
                            won = True;
                        else:
                            s = self.state_map[(x_next,y_next), (x_m,y_m)];
        
        if method == 'ValIter' or method == 'Qlearning' or method == 'sarsa':
            # Initialize current state, next state and time
            t = 1;
            state = self.map[start];
            # Add the starting position in the maze to the path
            path.append(start);

            # Add staring position of the minotaur to the minotaur path
            minotaur_path.append(minotaur_start);
            minotaur_state = self.minotaur_map[minotaur_start];

            if key:
                key_val = 0;
                s = self.state_map[(start,minotaur_start,key_val)];
            else:
                s = self.state_map[(start,minotaur_start)];


            # Loop while state is not the lost/won state
            while not lost_won:
                # Move to next state given the policy and the current state
                state, (x_next,y_next) = self.__move(state,policy[s]);
                # Add the position in the maze corresponding to the next state to the path
                path.append((x_next,y_next));

                minotaur_state, (x_m,y_m) = self.__minotaur_random_move(s,key);
                minotaur_path.append((x_m,y_m));


                # Update time and state for next iteration
                t +=1;
                # lost
                if (x_next == x_m and y_next == y_m):
                    s = self.state_map['lost'];
                    lost_won = True;
                    lost = True;
                else:
                    if key:
                        if (self.maze[x_next,y_next] == 2 and key_val): #won with key
                            s = self.state_map['won'];
                            lost_won = True;
                            won = True;
                        else:
                            if self.maze[(x_next,y_next)]==3:
                                key_val = 1;
                                # self.maze[(x_next,y_next)]=0; # since we got the key the cell is resetted
                            s = self.state_map[(x_next,y_next), (x_m,y_m), key_val];
                    
                    else: 
                        if (self.maze[x_next,y_next] == 2): #won
                            s = self.state_map['won'];
                            lost_won = True;
                            won = True;
                        else:
                            s = self.state_map[(x_next,y_next), (x_m,y_m)];
             
        return path, minotaur_path, lost, won
    
    
    def Q_learning_greedy(self, start, minotaur_start, key_cell, alpha_val = 2/3, gamma = 49/50, epsilon = 0.5, episodes = 1000, key = True, max_steps = 100):
        '''
            Returns a Q-table containing Q(s,a) defining the estimated optimal policy and the policy
        '''
        
        # value fn of the initial state
        vf_initial = list()

        # the rewards are needed
        r = self.minotaur_rewards;

        # 1. initialize Q
        n_states  = self.tot_states;
        n_actions = self.n_actions;

        Q   = np.zeros((n_states, n_actions));
        policy = np.zeros(n_states)
        # Initialize # visits
        n_visits = np.zeros((n_states,n_actions))
        
        for e in range(episodes):
            # 0. initialize S
            terminal = False; #terminal state

            key_val = 0
            s = self.state_map[start,minotaur_start, key_val]
            state = self.map[start]
            minotaur_state = self.minotaur_map[minotaur_start]
            self.maze[key_cell]=3; # the key is present at the beginning
            
            # Initialize time
            t = 1;
        
            # path = list();
            # minotaur_path = list();
            # path.append(state)
            # minotaur_path.append(minotaur_state)

            # Loop while state is not terminal, simulate an episode
            while not terminal and t < max_steps:
                # select an action epsilon-greedily
                # uniformly select any element from the list 
                n = random.uniform(0, 1);
                # epsilon-greedy part
                if n < epsilon: # select random action from the action space
                    action = random.randint(0, n_actions-1);
                else: # select best action
                    action = int(np.argmax(Q[s,:]))

                # Increment # of visits of pair [s,action]
                n_visits[s,action] += 1

                #update learning rate
                alpha = 1/(n_visits[s,action]**alpha_val)

                # take action A then observe the reward and the next state s'
                next_move, (x_next,y_next) = self.__move(state, action)
                # path.append((x_next,y_next));
                minotaur_next, (x_m,y_m) = self.__minotaur_random_move(s, key)
                # minotaur_path.append((x_m,y_m));

                # Update time and state for next iteration
                t +=1;
                # lost
                if (x_next == x_m and y_next == y_m):
                    s_next = self.state_map['lost'];
                    terminal = True;
                else:
                    if key:
                        if (self.maze[x_next,y_next] == 2 and key_val): #won with key
                            s_next = self.state_map['won'];
                            terminal = True;
                        else:
                            if self.maze[(x_next,y_next)]==3:
                                key_val = 1;
                                #self.maze[(x_next,y_next)]=0;
                            s_next = self.state_map[(x_next,y_next), (x_m,y_m), key_val];
                    
                    else: 
                        if (self.maze[x_next,y_next] == 2): #won
                            s_next = self.state_map['won'];
                            terminal = True;
                        else:
                            s_next = self.state_map[(x_next,y_next), (x_m,y_m)];
                
                
                # update Q function based on S and S'
                Q[s,action] = Q[s,action] + alpha * (r[s,action] + gamma * np.max(Q[s_next,:])-Q[s,action])

                s = s_next;
                state = next_move;
                minotaur_state = minotaur_next;

            # update value_fn of the initial state
            vf_initial.append(np.max(Q[self.state_map[start,minotaur_start,0],:]))
        
        policy = [np.argmax(Q[s,:]) for s in range(n_states)]
        return Q, policy, vf_initial
    
    def epsilon_greedy_action(self, epsilon, Q, s):
        n_actions = self.n_actions;
        # select an action epsilon-greedily
        # uniformly select any element from the list 
        n = random.uniform(0, 1);
        # epsilon-greedy part
        if n < epsilon: # select random action from the action space
            action = random.randint(0, n_actions-1);
        else: # select best action
            action = int(np.argmax(Q[s,:]))

        return action

    def sarsa(self, start, minotaur_start, key_cell, alpha_val = 2/3, gamma = 49/50, epsilon = 0.5, episodes = 1000, key = True, decreasing_epsilon = False,  delta = 2/3, max_steps = 150):
        # value fn of the initial state
        vf_initial = list()

        # the rewards are needed
        r = self.minotaur_rewards;

        # 1. initialize Q
        n_states  = self.tot_states;
        n_actions = self.n_actions;

        Q   = np.zeros((n_states, n_actions));
        policy = np.zeros(n_states)
        # Initialize # visits
        n_visits = np.zeros((n_states,n_actions))
        
        for e in range(episodes):
            # 0. initialize S
            terminal = False; #terminal state

            key_val = 0
            s = self.state_map[start,minotaur_start, key_val]
            state = self.map[start]
            self.maze[key_cell]=3; # the key is present at the beginning
            
            # Initialize time
            t = 1;

            if decreasing_epsilon:
                epsilon = 1/(e+1)**delta

            # OBSERVATIONS
            action = self.epsilon_greedy_action(epsilon, Q, s);

            # Loop while state is not terminal, simulate an episode
            while not terminal and t < max_steps:
                # Increment # of visits of pair [s,action]
                n_visits[s,action] += 1

                #update learning rate
                alpha = 1/(n_visits[s,action]**alpha_val)

                # take action A then observe the reward and the next state s'
                next_move, (x_next,y_next) = self.__move(state, action)
                _, (x_m,y_m) = self.__minotaur_random_move(s, key)

                # lost
                if (x_next == x_m and y_next == y_m):
                    s_next = self.state_map['lost'];
                    terminal = True;
                else:
                    if key:
                        if (self.maze[x_next,y_next] == 2 and key_val): #won with key
                            s_next = self.state_map['won'];
                            terminal = True;
                        else:
                            if self.maze[(x_next,y_next)]==3:
                                key_val = 1;
                                #self.maze[(x_next,y_next)]=0; # since we got the key the cell is resetted
                            s_next = self.state_map[(x_next,y_next), (x_m,y_m), key_val];
                    
                    else: 
                        if (self.maze[x_next,y_next] == 2): #won
                            s_next = self.state_map['won'];
                            terminal = True;
                        else:
                            s_next = self.state_map[(x_next,y_next), (x_m,y_m)];
                
                next_action = self.epsilon_greedy_action(epsilon, Q, s_next);
                
                # update Q function based on S and S'
                Q[s,action] = Q[s,action] + alpha * (r[s,action] + gamma * Q[s_next,next_action]-Q[s,action])

                

                s = s_next;
                state = next_move;
                action = next_action;
            
                # Update time and state for next iteration
                t +=1;

            # update value_fn of the initial state
            vf_initial.append(np.max(Q[self.state_map[start,minotaur_start,0],:]))
        
        policy = [np.argmax(Q[s,:]) for s in range(n_states)]
        return Q, policy, vf_initial


    def show(self):
        print('The states are :')
        print(self.states)
        print('The actions are:')
        print(self.actions)
        print('The mapping of the states:')
        print(self.map)
        print('The rewards:')
        print(self.minotaur_rewards)

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

def draw_policy(env, minotaur_position, policy, time, key = False, key_val = 0):
    maze = env.maze;
    (x_m,y_m) = minotaur_position;
    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, 3:YELLOW, -6: LIGHT_RED, -1: LIGHT_RED};

    # Give a color to each cell
    rows,cols    = maze.shape;
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows));

    # Remove the axis ticks and add title title
    ax = plt.gca();
    if time == -1:
        ax.set_title('Optimal Policy');
    else: 
        ax.set_title('Optimal Policy at time t =%i' %time);
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
                if key:
                    state = env.state_map[(x,y),(x_m,y_m), key_val];
                else:
                    state = env.state_map[(x,y),(x_m,y_m)];
                if time == -1:
                    action = policy[state];
                else:
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

def value_iteration_minotaur(env, gamma, epsilon):
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
    r         = env.minotaur_rewards;
    n_states  = env.tot_states;
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
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, 3: YELLOW, -6: LIGHT_RED, -1: LIGHT_RED};

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
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, 3: YELLOW, -6: LIGHT_RED, -1: LIGHT_RED};

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

            # Minotaur
            grid.get_celld()[(minotaur_path[i])].set_facecolor(LIGHT_PURPLE)
            grid.get_celld()[(minotaur_path[i])].get_text().set_text('Minotaur')

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


def compute_wining_probability(env, start=(0,0), minotaur_start=(6,5), runs=10000, horizon=30 , method='DynProg'):
        '''
        Computes the winning probability based on the number of runs and the horizon
        :return list of probabilities for each horizon
        '''
        probabilities = []

        for h in range(1, horizon+1):
            V, policy = dynamic_programming_minotaur(env, h)
            wins = 0
            # for i in range(runs):
            #     _, _, _, won = env.simulate_minotaur(start, minotaur_start, policy, 'DynProg')
            #     if won:
            #         wins += 1
            # 
            # prob = wins / runs
            prob = get_probability(env, policy, start, minotaur_start, runs, method=method)
            probabilities.append(prob)
        return probabilities

def get_probability(env, policy, start=(0,0), minotaur_start=(6,5), runs=10000, method = 'DynProg'):
    '''
    This function gives you the probability of winning given a policy, method and number of runs
    :return probability
    '''
    wins=0
    for i in range(runs):
        _, _, _, won = env.simulate_minotaur(start, minotaur_start, policy, method)
        if won:
            wins += 1
                
    return  wins / runs
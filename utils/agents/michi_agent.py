from michi.michi import *


class MichiAgent(object):

    def __init__(self, n_sims):

        self.n_sims = n_sims


    def tensor_to_board(self, state):
        

        tree = TreeNode(pos=empty_position())
        tree.expand()
        owner_map = W*W*[0]


        return tree


    def act(self, state):


        tree = self.tensor_to_board(state)

        owner_map = W*W*[0]
        tree = tree_search(tree, N_SIMS, owner_map)



def main():
    
    michi_agent = MichiAgent()

    michi_agent.act(None)

if __name__ == "__main__":

    main()
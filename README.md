# AudioGenAI

### parameters:
1. Generate audio using Reinforcement Learning SARSA algorithm:

        "GEN" "RL" "SARSA" "the power of 10 for the horizon" "path to reference file"


2. Fix audio using Reinforcement Learning SARSA algorithm (can also use 'SARSA_LAMBDA' instead of SARSA):
   
          "FIX" "RL" "SARSA" "path to reference file" "the power of 10 for the horizon" "path to file to fix" "path to correct file"

3. Export "single note audio" from normal audio (currently supported: 'NO_ERRORS', '1'):

            "SINGLE_NOTE" "ERROR_TYPE" "path to source file"

4. Calculating statistics (can also use 'SARSA_LAMBDA' instead of SARSA):

             "STATISTICS" "RL" "SARSA" "path to reference file" "the power of 10 for the horizon" "path to file to fix" "path to correct file"
5. Finding good (w.r to MSE) Q function (can also use 'SARSA_LAMBDA' instead of SARSA):

             "FIND_Q" "RL" "SARSA" "path to reference file" "the power of 10 for the horizon" "path to file to fix" "path to correct file" "num_of_exploration_iterations"
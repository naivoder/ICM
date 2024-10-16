# Intrinsic Curiosity Module (ICM)

🚧👷🛑 Under Construction!!!

## Table of Contents

1. [Overview](#overview)
2. [Setup](#setup)
3. [Results](#results)
4. [Analysis](#analysis)

## Overview

Note to self: different values of T_MAX have a significant effect for different environments...  

This repository contains an implementation of the Intrinsic Curiosity Module (ICM) built upon an A3C (Asynchronous Advantage Actor-Critic) backend using PyTorch. The ICM encourages exploration by predicting an agent's next state given its current state and action, generating an intrinsic reward that complements extrinsic rewards from the environment. The algorithm is evaluated on various Atari environments using Gymnasium.

## Setup

### Required Dependencies

It's recommended to use a Conda environment to manage dependencies and avoid conflicts. You can create and activate a new Conda environment with the following commands:

```bash
conda create -n rl python=3.10
conda activate rl
```

After activating the environment, install the required dependencies using:

```bash
pip install -r requirements.txt
```

### Running the Algorithm

You can run the ICM-enhanced A3C algorithm on any supported Gymnasium Atari environment with a discrete action space using the following command:

```bash
python main.py --env 'MsPacmanNoFrameskip-v4'
```

- **Environment Selection**: Use `-e` or `--env` to specify the Gymnasium environment. The default is `None`, so you must specify an environment.
  
  Example:

  ```bash
  python main.py --env 'PongNoFrameskip-v4'
  ```

- **Number of Training Episodes**: Use `--n_games` to specify the number of games the agent should play during training.

  Example:

  ```bash
  python main.py --n_games 5000
  ```

- **Parallel Environments**: Use `--n_envs` to specify the number of parallel environments to run during training. The default is 4.

  Example:

  ```bash
  python main.py --env 'AsterixNoFrameskip-v4' --n_envs 16
  ```

Using a Conda environment along with these flexible command-line options will help you efficiently manage your dependencies and customize the training process for your specific needs.

## Results

<table>
    <tr>
        <td>
            <p><b>AirRaid</b></p>
            <img src="environments/AirRaidNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Alien</b></p>
            <img src="environments/AlienNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Amidar</b></p>
            <img src="environments/AmidarNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="metrics/AirRaidNoFrameskip-v4_learning_curve.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/AlienNoFrameskip-v4_learning_curve.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/AmidarNoFrameskip-v4_learning_curve.png" width="250" height="250"/>
        </td>
    </tr>
</table>
<table>
    <tr>
        <td>
            <p><b>Assault</b></p>
            <img src="environments/AssaultNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Asterix</b></p>
            <img src="environments/AsterixNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Asteroids</b></p>
            <img src="environments/AsteroidsNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="metrics/AssaultNoFrameskip-v4_learning_curve.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/AsterixNoFrameskip-v4_learning_curve.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/AsteroidsNoFrameskip-v4_learning_curve.png" width="250" height="250"/>
        </td>
    </tr>
</table>
<table>
    <tr>
        <td>
            <p><b>Atlantis</b></p>
            <img src="environments/AtlantisNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>BankHeist</b></p>
            <img src="environments/BankHeistNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>BattleZone</b></p>
            <img src="environments/BattleZoneNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="metrics/AtlantisNoFrameskip-v4_learning_curve.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/BankHeistNoFrameskip-v4_learning_curve.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/BattleZoneNoFrameskip-v4_learning_curve.png" width="250" height="250"/>
        </td>
    </tr>
</table>
<table>
    <tr>
        <td>
            <p><b>BeamRider</b></p>
            <img src="environments/BeamRiderNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Berzerk</b></p>
            <img src="environments/BerzerkNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Bowling</b></p>
            <img src="environments/BowlingNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="metrics/BeamRiderNoFrameskip-v4_learning_curve.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/BerzerkNoFrameskip-v4_learning_curve.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/BowlingNoFrameskip-v4_learning_curve.png" width="250" height="250"/>
        </td>
    </tr>
</table>
<table>
    <tr>
        <td>
            <p><b>Boxing</b></p>
            <img src="environments/BoxingNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Breakout</b></p>
            <img src="environments/BreakoutNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Carnival</b></p>
            <img src="environments/CarnivalNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="metrics/BoxingNoFrameskip-v4_learning_curve.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/BreakoutNoFrameskip-v4_learning_curve.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/CarnivalNoFrameskip-v4_learning_curve.png" width="250" height="250"/>
        </td>
    </tr>
</table>
<table>
    <tr>
        <td>
            <p><b>Centipede</b></p>
            <img src="environments/CentipedeNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>ChopperCommand</b></p>
            <img src="environments/ChopperCommandNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>CrazyClimber</b></p>
            <img src="environments/CrazyClimberNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="metrics/CentipedeNoFrameskip-v4_learning_curve.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/ChopperCommandNoFrameskip-v4_learning_curve.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/CrazyClimberNoFrameskip-v4_learning_curve.png" width="250" height="250"/>
        </td>
    </tr>
</table>
<table>
    <tr>
        <td>
            <p><b>Defender</b></p>
            <img src="environments/DefenderNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <!--<td>
            <p><b>ChopperCommand</b></p>
            <img src="environments/ChopperCommandNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>CrazyClimber</b></p>
            <img src="environments/CrazyClimberNoFrameskip-v4.gif" width="250" height="250"/>
        </td>-->
    </tr>
    <tr>
        <td>
            <img src="metrics/DefenderNoFrameskip-v4_learning_curve.png" width="250" height="250"/>
        </td>
        <!--<td>
            <img src="metrics/ChopperCommandNoFrameskip-v4_learning_curve.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/CrazyClimberNoFrameskip-v4_learning_curve.png" width="250" height="250"/>
        </td>-->
    </tr>
</table>

## Analysis

<!--
## Analysis of Results

This section will explore the performance of the ICM module across various Atari games, focusing on its ability to drive exploration in environments with sparse rewards. We will compare the performance of ICM-enhanced A3C with standard A3C to highlight the effectiveness of intrinsic rewards.
-->

## Acknowledgements

Shout out to Phil Tabor, an excellent teacher! I highly recommend his [Youtube channel](https://www.youtube.com/machinelearningwithphil).

Special thanks to Deepak Pathak and his team for their work. Check out the original paper and their project [here](https://pathak22.github.io/noreward-rl/).

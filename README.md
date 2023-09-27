<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]
-->



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/M-Hardy/MLFS">
    <img src="images/repo_icon.jpg" alt="Logo" width="80" height="80">
  </a>

<h3 align="center">Machine Learning From Scratch (MLFS)</h3>

  <p align="center">
    Machine learning model implementations without machine learning libraries.
    <!-- 
    <br />
    <a href="https://github.com/M-Hardy/MLFS"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/M-Hardy/MLFS">View Demo</a>
    ·
    <a href="https://github.com/M-Hardy/MLFS/issues">Report Bug</a>
    ·
    <a href="https://github.com/M-Hardy/MLFS/issues">Request Feature</a>
    -->
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <!--
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    -->
    <li><a href="#latest-model">Latest Model</a></li>
      <ul>
        <li><a href="#mnist-model">MLP for Hand-Written Digit Recognition</a></li>
      </ul>
    <li><a href="#roadmap">Roadmap</a></li>
    <!--
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    -->
    <li><a href="#contact">Contact</a></li>
    <!--
    <li><a href="#acknowledgments">Acknowledgments</a></li>
    -->
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

MLFS is a personal project focused on building machine learning models from scratch to deepen understanding and avoid relying on pre-made abstractions. 

Implementations are in Python, use of machine learning libraries is banned (pytorch, tensorflow, jax, etc.). Numpy and matplotlib libraries are used for vectorization and plotting subroutines. 

The repo is divided into modules encompassing:
1. **Data handling:** Subroutines for loading data, feature scaling, and data splitting
2. **Model implementations (as of 2023-09-26):** Linear regression, logistic regression, MLP (MNIST)
4. **Plotting:** Routines for plotting cost/accuracy over training iterations using saved model metadata
5. **Model IO:** Routines to save/load model metadata and plots in auto-generated folders
   
Repo is WIP. Machine learning model implementations will be added and updated as my learning progresses.

<!-- [![Product Name Screen Shot][product-screenshot]](https://example.com) -->

<!-- Here's a blank template to get started: To avoid retyping too much info. Do a search and replace with your text editor for the following: `github_username`, `repo_name`, `twitter_handle`, `linkedin_username`, `email_client`, `email`, `project_title`, `project_description` -->

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With
<!--
* [![Next][Next.js]][Next-url]
* [![React][React.js]][React-url]
* [![Vue][Vue.js]][Vue-url]
* [![Angular][Angular.io]][Angular-url]
* [![Svelte][Svelte.dev]][Svelte-url]
* [![Laravel][Laravel.com]][Laravel-url]
* [![Bootstrap][Bootstrap.com]][Bootstrap-url]
* [![JQuery][JQuery.com]][JQuery-url]
-->
[![Python][Python.com]][Python-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED 
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* npm
  ```sh
  npm install npm@latest -g
  ```

### Installation

1. Get a free API Key at [https://example.com](https://example.com)
2. Clone the repo
   ```sh
   git clone https://github.com/M-Hardy/MLFS.git
   ```
3. Install NPM packages
   ```sh
   npm install
   ```
4. Enter your API in `config.js`
   ```js
   const API_KEY = 'ENTER YOUR API';
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>
-->


<!-- USAGE EXAMPLES 
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>
-->

<!-- Latest Model -->
## Latest Model: MLP for MNIST Hand-Written Digit Recognition

The MNIST model is a 3-layer multilayer perceptron, consisting of 50, 25, and 10 nodes respectively, trained on the MNIST handwritten digit dataset. 
* The dataset consists of 70,000 images, each 28x28 pixels in size
* Each pixel in the image is represented by a value  indicating the grayscale intensity of the pixel, resulting in 784 features/training example
* The dataset used to train the model can be found here: https://data.world/nrippner/mnist-handwritten-digits 

The MNIST model implementation has 4 basic elements: forward propagation, back propagation, and main functions to train the model and test different learning rates. 

### 1. Forward Propagation
Forward-propagation is implemented using dense_layer functions, which take input data, weight & bias parameters, and an activation function, and returns the output of the activation function applied to the logits of the input data. ReLu and softmax activation functions - for the hidden layers and the output layer respectively - are defined. 

Sequentially passing the output of a layer function into another layer function emulates a feedfoward network, which is implemented by wrapping layer function calls in a forward_prop function.

### 2. Back Propagation

Gradients with respect to the logits, weights, and bias parameters for each layer of the model are computed explicitly to conduct back propagation. Subroutines for one-hot encoding the targets and computing the derivative of the ReLu activation function are used in the computation. 

Cost and accuracy of the model are periodically saved during gradient descent to measure the performance/precision of the model over training iterations. 


### 3. Main Function: Run Model

Data is loaded, scaled using z-score normalization, and split into training and cross-validation sets. Weight and bias parameter values are randomly initialized for each layer in the neural network, and then the model is trained using gradient descent.  

In addition to the cost and accuracy metrics recorded during training, additional model metadata is recorded upon training completion: final weight and bias parameter values, learning rate, number of units in each layer, number of training iterations, etc. 


### 4. Main Function: Test Learning Rates

Currently, the most straightforward way to find a good learning rate for the MNIST model (via MLFS) is to train multiple models with the same architecture using different learning rates. The metadata for each model is saved, and the cost/iterations and accuracy/iterations of each model is plotted using plotting routines defined in another module in the repo. This gives a visualization of how the MNIST model performs using different learning rates.

Implementing an optimization algorithm that automatically updates the learning rate during training (e.g. Adam optimizer) is a natural next step to reduce the current overhead in finding an efficient learning rate.

## Evaluating the MNIST Model

- Static cost problem
- Cost/accuracy images
- Indications of best learning rate
- CV set performance = good generalization, no overfitting on training set


<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- ROADMAP -->
## Roadmap
- [ ] Add regularization to protect against overfitting
- [ ] Add batch processing to improve training speed
- [ ] Improve model plots
- [ ] Create config file(s) for all hyperparameters/constants in repo
- [ ] Create a plotting routine that plots decision boundaries of classification models
- [ ] Implement an optimization algorithm for learning rate during gradient descent (e.g. Adam optimizer)
- [ ] Create a model-agnostic run_model() script in the main project dir that allows you to load in a model & data, and train & test it
- [ ] Create an autograd engine - avoid explicitly computing gradients for each neural net you implement
- [ ] MNIST model:
    - [ ] Add hyperparameter for recording model cost/accuracy during gradient descent at Xth iterations
    - [ ] Create subroutine to cast dataset to int (nripper dataset has all values as floats)
    - [ ] Refactor create_train_and_cv_set to take individual x and y arguments as opposed to splitting them from a single matrix within the function
  


<!-- See the [open issues](https://github.com/M-Hardy/MLFS/issues) for a full list of proposed features (and known issues). -->

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING 
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>
-->


<!-- LICENSE 
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>
-->



<!-- CONTACT -->
## Contact

Your Name - [@twitter_handle](https://twitter.com/twitter_handle) - email@email_client.com

Project Link: [https://github.com/github_username/repo_name](https://github.com/M-Hardy/MLFS)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS 
## Acknowledgments

* []()
* []()
* []()

<p align="right">(<a href="#readme-top">back to top</a>)</p>
-->


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
<!--
[contributors-shield]: https://img.shields.io/github/contributors/M-Hardy/MLFS.svg?style=for-the-badge
[contributors-url]: https://github.com/M-Hardy/MLFS/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/M-Hardy/MLFS.svg?style=for-the-badge
[forks-url]: https://github.com/M-Hardy/MLFS/network/members
[stars-shield]: https://img.shields.io/github/stars/M-Hardy/MLFS.svg?style=for-the-badge
[stars-url]: https://github.com/M-Hardy/MLFS/stargazers
[issues-shield]: https://img.shields.io/github/issues/M-Hardy/MLFS.svg?style=for-the-badge
[issues-url]: https://github.com/M-Hardy/MLFS/issues
[license-shield]: https://img.shields.io/github/license/M-Hardy/MLFS.svg?style=for-the-badge
[license-url]: https://github.com/M-Hardy/MLFS/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/michael-b-hardy/
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 
-->
[Python.com]: https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54
[Python-url]: https://www.python.org/

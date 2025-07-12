<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->


<a id="readme-top"></a>
<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/art-kors/StyleGAN-NADA">
    <img src="images/logo.jfif" alt="Logo" width="80" height="80">
  </a>

<h3 align="center">StyleGAN-NADA Reimplementation</h3>

  <p align="center">
    Reimplementation of StyleGAN-NADA with simple architecture. 
    <br />
    <a href="https://stylegan-nada.github.io/"><strong>Original »</strong></a>
    <br />
    <br />
    <a href="https://github.com/github_username/repo_name">View Demo</a>
    &middot;
    <a href="https://github.com/art-kors/StyleGAN-NADA/issues/new">Report Bug</a>
    &middot;
  </p>
</div>

<!-- ABOUT THE PROJECT -->
## About The Project

<img src="/images/architecture.png" alt="contrib.rocks image" />


This project is a reimplementation of [StyleGAN-NADA](https://stylegan-nada.github.io/). In my work, I omitted quite a lot of details made in the original implementation, but this work will be easier to understand, because it does not have any additional features, such as video generation or launch from Docker.


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

### Requirements

- Anaconda


### Pretrained model

you can download weights in `.pth` format [here.](https://drive.google.com/file/d/1bhWgbI7oleIBPs9kE7IL7LGkZpzNOsFO/view?usp=sharing)



### Train your own model

1. Open the `stylegan_nada.ipynb` locally or colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/rinongal/stylegan-nada/blob/main/stylegan_nada.ipynb) 
2. Enter the text description of style A and style B
3. Specify the path to the image you want to convert and the path to save it.
4. press "Run all". In process you need login in google drive to download StyleGAN2 weights, it's okay.
5. Look at results.

<p align="right">(<a href="#readme-top">back to top</a>)</p>




## Report

read the REPORT.md
<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

Artemii Korsaev - [@AK_N0maD](https://t.me/AK_N0maD) - art.kors@yandex.ru
<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [Deep Learning School](https://dls.samcs.ru/)
* [Nina Konovalova](https://t.me/reading_ai)
* My parents

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[PyTorch-shield]: https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white
[PyTorch-url]: https://pytorch.org/
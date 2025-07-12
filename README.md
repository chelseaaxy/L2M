# Learning Dense Feature Matching via Lifting Single 2D Image to 3D Space

![L2M Logo](https://img.shields.io/badge/L2M-Official%20Implementation-blue)

Welcome to the **L2M** repository! This is the official implementation of our ICCV'25 paper titled "Learning Dense Feature Matching via Lifting Single 2D Image to 3D Space". This project focuses on enhancing feature matching techniques in computer vision by leveraging 3D space representation.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview

In this repository, we present a novel approach to dense feature matching. Our method lifts a single 2D image into 3D space, allowing for improved feature extraction and matching. The approach aims to address the limitations of traditional 2D feature matching techniques, particularly in complex scenes.

### Key Features

- **3D Representation**: Transform 2D images into 3D space for better feature matching.
- **State-of-the-art Performance**: Achieve superior results compared to existing methods.
- **Open-source**: Freely available for research and development.

## Installation

To get started, clone this repository to your local machine:

```bash
git clone https://github.com/chelseaaxy/L2M.git
cd L2M
```

### Dependencies

Make sure to install the required dependencies. You can do this using pip:

```bash
pip install -r requirements.txt
```

## Usage

To use the L2M implementation, you need to download the latest release from our [Releases section](https://github.com/chelseaaxy/L2M/releases). After downloading, follow these steps:

1. Extract the downloaded files.
2. Run the main script:

```bash
python main.py --input <path_to_your_image>
```

Replace `<path_to_your_image>` with the path to the image you want to process.

## Dataset

For training and testing, we utilized several datasets that are commonly used in feature matching tasks. Here are the main datasets:

- **KITTI**: A popular dataset for autonomous driving applications.
- **Oxford**: A dataset containing various scenes and landmarks.
- **Aerial**: High-resolution aerial images for feature matching.

You can download these datasets from their respective sources and place them in the `data` directory.

## Results

Our method has shown significant improvements in feature matching tasks. Below are some examples of the results achieved using our approach:

![Result Example 1](https://example.com/result1.png)
![Result Example 2](https://example.com/result2.png)

For detailed results and benchmarks, refer to the results section in our paper.

## Contributing

We welcome contributions to this project! If you have suggestions, bug fixes, or new features, please feel free to open an issue or submit a pull request. 

### Steps to Contribute

1. Fork the repository.
2. Create a new branch for your feature or fix.
3. Make your changes and commit them.
4. Push your branch and create a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact

For any inquiries, please reach out to us via email or check our [Releases section](https://github.com/chelseaaxy/L2M/releases) for updates.

---

Thank you for your interest in L2M! We look forward to your contributions and feedback.
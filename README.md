# SkySpectral
A simple tool to process multispectral images from the Micasense RedEdge-P

Coming next:
- Pan sharpening algorithm
- Agisoft automated workflow
- Image annotation

## Introduction
SkySpectral is a simple tool for visualizing and processing multispectral imagery. It enables the alignment and combination of various sub-images (bands) to extract relevant data.

\#Multispectral \#Open-CV \#Building diagnosis \#ImageProcessing 

**The project is still in pre-release, so do not hesitate to send your recommendations or the bugs you encountered!**

<p align="center">
    <a><img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExeDJkdnhmaGJqaXE1OHBwOGYzb3Y2bjNlbnp3ZmN3aGU3bmhoZGoydiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/PhPLPMBCVX929LJflR/giphy.gif" alt="principle" border="0"></a>
    
    Processing multispectral image, from a folder
</p>


## Principle
The functionalities include:
- Import Micasense output folder
- Visualize bands with standard matplotlib color palettes
- Align channels (R,G,B,NIR,Red-Edge) to Pan-chromatic with an intuitive interface (especially critical for close-shots, as in building façades analyses)
- Visualize composed shot (RGB image re-composed from individual channels, CIR - Color Infrared, etc.)
- Create custom indices based on arithmetic operations on bands
- Access typical vegetation indices such as NDVI
- Prepare the files for Agisoft Metashape (organize images in folders)
- ...

<p align="center">
    <a href="https://ibb.co/Dg8Rm1g"><img src="https://i.ibb.co/h1Zmrg1/Capture-d-cran-2023-09-29-095333.png" alt="Capture-d-cran-2023-09-29-095333" border="0"></a>
    
    Creating custom indices
</p>



## Installation instructions

1. Clone the repository:
```
git clone https://github.com/s-du/SkySpectral
```

2. Navigate to the app directory:
```
cd SkySpectral
```

3. Install the required dependencies:
```
pip install -r requirements.txt
```

4. Run the app:
```
python main.py
```

## Contributing

Contributions to the app are welcome! If you find any bugs, have suggestions for new features, or would like to contribute enhancements, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make the necessary changes and commit them.
4. Push your changes to your fork.
5. Submit a pull request describing your changes.

# AprilTag Detection Pipeline

This project includes a class that provides AprilTag detection functionality for a FIRST Tech Challenge (FTC) robot. It uses the OpenCV and EasyOpenCV libraries to detect AprilTags in the robot's surroundings and determine their positional information.

## Table of Contents
- [Libraries](#libraries)
- [Class Definition](#class-definition)
- [Features](#features)
- [Constructor](#constructor)
- [Main Functions](#main-functions)
- [Image Processing](#image-processing)
- [Goal](#goal)

## Libraries
- `org.opencv.*`: Core functions of OpenCV.
- `org.openftc.apriltag.*`: Library used for AprilTag detection.
- `org.opencv.imgproc.*`: Functions for image processing.

## Class Definition
### AprilTagDetectionPipeline
The main class that performs AprilTag detection, derived from the `OpenCvPipeline` class.

## Features
- **tagDistances**: Stores the distances of detected tags.
- **currentTagIDs**: Stores the currently detected tag IDs.
- **telemetry**: Updates telemetry data.

## Constructor
Creates an `AprilTagDetectionPipeline` object by taking tag size, camera parameters, and a telemetry object.

## Main Functions
- **processFrame(Mat input)**: Processes the input image to detect tags. 
- **drawAxisMarker**: Draws a 3D axis marker.
- **draw3dCubeMarker**: Draws a 3D cube marker.
- **poseFromTrapezoid**: Used to determine the pose of the tags.
- **sortPoints**: Sorts points in a clockwise direction.

## Image Processing
- The input image is converted to grayscale.
- The AprilTag detection function is called to identify detected tags and calculate their distances.
- Telemetry is updated for detected tags, and notifications are made for lost tags.

## Goal
The purpose of the code is to detect the AprilTags in the robot's surroundings, determine the positions of these tags, and enable the robot to move more accurately based on this information.

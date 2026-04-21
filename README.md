# Real-Time Hand Gesture Recognition System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-orange.svg)](https://mediapipe.dev/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> A real-time computer vision system that detects and recognizes 20+ hand gestures using webcam input. Built with Python, OpenCV, and MediaPipe.

# Table of Contents
- [Features](#-features)
- [Recognized Gestures](#-recognized-gestures)
- [Tech Stack](#️-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [How It Works](#-how-it-works)
- [Performance](#-performance)
- [Troubleshooting](#-troubleshooting)
- [Roadmap](#️-roadmap)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

# Features

- Real-time hand tracking - 30+ FPS performance
- 21 landmark points per hand with MediaPipe
- 20+ gesture recognition including numbers, letters, shapes, and actions
- Left/Right hand detection with orientation-specific logic
- Multi-hand support - track 2 hands simultaneously
- Gesture smoothing - prevents flickering between frames
- Visual feedback - bounding boxes, finger states, and info panel
- Screenshot capture - save your gestures with 's' key
- FPS counter - monitor performance in real-time

# Recognized Gestures

# Number Gestures
| Gesture | Description | Output |
|---------|-------------|---------|
| 👊 | Fist | ZERO (0) |
| ☝️ | Index finger up | ONE (1) |
| ✌️ | Index & middle up | TWO (2) |
| 🤟 | Index, middle, ring up | THREE (3) |
| 🤘 | Four fingers (no thumb) | FOUR (4) |
| 🖐️ | All fingers open | FIVE (5) |
| 👍 | Thumb only | SIX (6) |

# Letter Gestures (ASL Inspired)
| Gesture | Output |X
|---------|---------|
| ✊ | LETTER A |
| 🖐️ | LETTER B |
| 🇨 | LETTER C |
| 🇱 | LETTER L |
| 👌 | LETTER O / OK |
| ✌️ | LETTER V / PEACE |

# Shape & Symbol Gestures
| Gesture | Output |
|---------|---------|
| ❤️ | HEART |
| ⭐ | STAR / SPIDER |
| 🔫 | GUN |
| 📞 | CALL ME |
| 🤘 | ROCK ON |
| 🕷️ | SPIDER-MAN |

# Control Gestures
| Gesture | Output |
|---------|---------|
| 🤏 | PINCH |
| 👆 | POINTING |
| ✌️ | DOUBLE POINT |
| 🦞 | GRAB / CLAW |

# Emotion Gestures
| Gesture | Output |
|---------|---------|
| 👍 | THUMBS UP |
| 👎 | THUMBS DOWN |
| 🖐️ | HIGH FIVE |
| 👊 | FIST BUMP |

# Tech Stack

| Technology | Purpose |
|------------|---------|
| Python 3.8+ | Core programming language |
| OpenCV | Webcam capture, image processing, display |
| MediaPipe | Hand landmark detection (21 points per hand) |
| NumPy | Mathematical operations and coordinate calculations |
| Collections | Gesture smoothing with Counter |

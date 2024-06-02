import shutil
import time

import streamlit as st
from PIL import Image
import pytesseract
import os

from app import rule_agent


# Set the path to the tessaract OCR engine
def ocr_image(image, doc_name):
    # Convert the image to grayscale
    gray_image = image.convert('L')

    # Perform OCR on the image
    text = pytesseract.image_to_string(gray_image)

    # Save the extracted text to a file
    with open(f"session/{doc_name}.txt", "w", encoding="utf-8") as file:
        file.write(text)

    return text

def sample_rule(rule_statement):
    # This is a sample rule function that always returns "yes"
    return "yes"

def erase_dir():
    if os.path.isdir("session"):
        shutil.rmtree("session")
    os.makedirs("session")

def rule_call(rule_statement):
    ans = []
    for rule in rule_statement.split("\n"):
        # res = rule_agent(rule)
        res = {"status":"no","reason":"hurray"}
        ans.append([rule,res.get("status","not found"),res.get("reason","")])
    return ans

def main():
    st.title("OCR Image to Text")
    erase_dir()

    # Upload images
    image1 = st.file_uploader("Upload Image 1", type=["png", "jpg", "jpeg"])
    doc_name1 = st.text_input("Enter Document Name for Image 1")

    image2 = st.file_uploader("Upload Image 2", type=["png", "jpg", "jpeg"])
    doc_name2 = st.text_input("Enter Document Name for Image 2")

    # Input rule statement
    rule_statement = st.text_area("Enter a rule statement")

    if st.button("Extract Text"):
        if image1 is not None and doc_name1:
            img1 = Image.open(image1)
            text1 = ocr_image(img1, doc_name1)
            st.success(f"Text extracted from {doc_name1} and saved as {doc_name1}.txt")
            st.write(f"Extracted Text: \n{text1}")

        if image2 is not None and doc_name2:
            img2 = Image.open(image2)
            text2 = ocr_image(img2, doc_name2)
            st.success(f"Text extracted from {doc_name2} and saved as {doc_name2}.txt")
            st.write(f"Extracted Text: \n{text2}")


        if rule_statement:
            for rule in rule_statement.split("\n"):
                st.success(f"Processing Rule statement : {rule}")
                res = rule_agent(rule)
                # res = {"status": "no", "reason": "hurray"}
                status = res.get("status","")
                reason = res.get("reason","")
                if status.lower().strip() == "yes":
                    st.success(f"{rule[1]},{reason}")
                elif status.lower().strip() == "no":
                    st.error(f"{rule[1]}, {reason}")
                else:
                    st.warning(f"Rule is not found")

if __name__ == "__main__":
    main()
import streamlit as st
import numpy as np

st.set_page_config(page_title="Coordinate Transformer", layout="centered")

st.title("Coordinate Transformation App")
st.write("Convert between Cartesian, Cylindrical, and Spherical coordinates easily.")

# ---------------- Utility Functions ----------------
def cart_to_cyl(x, y, z):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta, z

def cart_to_sph(x, y, z):
    rho = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y, x)
    phi = np.arccos(z / rho) if rho != 0 else 0
    return rho, theta, phi

def cyl_to_cart(r, theta, z):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y, z

def sph_to_cart(rho, theta, phi):
    x = rho * np.sin(phi) * np.cos(theta)
    y = rho * np.sin(phi) * np.sin(theta)
    z = rho * np.cos(phi)
    return x, y, z

def cyl_to_sph(r, theta, z):
    return cart_to_sph(*cyl_to_cart(r, theta, z))

def sph_to_cyl(rho, theta, phi):
    return cart_to_cyl(*sph_to_cart(rho, theta, phi))

# ---------------- UI ----------------
mode = st.selectbox(
    "Select transformation:",
    [
        "Cartesian → Cylindrical",
        "Cartesian → Spherical",
        "Cylindrical → Cartesian",
        "Spherical → Cartesian",
        "Cylindrical → Spherical",
        "Spherical → Cylindrical",
    ]
)

st.subheader("Input Values")
col1, col2, col3 = st.columns(3)

def show_inputs(labels):
    return [col1.number_input(labels[0], value=0.0),
            col2.number_input(labels[1], value=0.0),
            col3.number_input(labels[2], value=0.0)]

# ---------------- Compute Transformation ----------------
if mode == "Cartesian → Cylindrical":
    x, y, z = show_inputs(["x", "y", "z"])
    result = cart_to_cyl(x, y, z)
    result_labels = ["r", "θ (rad)", "z"]

elif mode == "Cartesian → Spherical":
    x, y, z = show_inputs(["x", "y", "z"])
    result = cart_to_sph(x, y, z)
    result_labels = ["ρ", "θ (rad)", "φ (rad)"]

elif mode == "Cylindrical → Cartesian":
    r, theta, z = show_inputs(["r", "θ (rad)", "z"])
    result = cyl_to_cart(r, theta, z)
    result_labels = ["x", "y", "z"]

elif mode == "Spherical → Cartesian":
    rho, theta, phi = show_inputs(["ρ", "θ (rad)", "φ (rad)"])
    result = sph_to_cart(rho, theta, phi)
    result_labels = ["x", "y", "z"]

elif mode == "Cylindrical → Spherical":
    r, theta, z = show_inputs(["r", "θ (rad)", "z"])
    result = cyl_to_sph(r, theta, z)
    result_labels = ["ρ", "θ (rad)", "φ (rad)"]

elif mode == "Spherical → Cylindrical":
    rho, theta, phi = show_inputs(["ρ", "θ (rad)", "φ (rad)"])
    result = sph_to_cyl(rho, theta, phi)
    result_labels = ["r", "θ (rad)", "z"]

# ---------------- Display Result ----------------
st.subheader("Result")
for label, val in zip(result_labels, result):
    st.write(f"{label} = {val:.4f}")

# ---------------- Download Code ----------------
import inspect
code = inspect.getsource(cart_to_cyl) + "\n# Add the rest of your functions here..."
st.download_button("Download Example Function", code, "coordinate_transform.py")

st.caption("Made with ♡ using Python + Streamlit")

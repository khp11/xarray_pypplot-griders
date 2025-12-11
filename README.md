# xarray_pypplot 
just using python all plots for bmd and sfd for grider
bending moment diagram and shear force diagram for  grider : seperate insight and plots for central grider for 2d plot and 3d plot for all grinder simultaneously an interactive 3d graph to have info just u hover at nodes : the 3d plots will be saved as .html in ur working directory and 2d plot as images



You can copy–paste directly into GitHub.

read more for full details...

---

# 📘 **README.md — 3D SFD & BMD Visualization (Plotly + Xarray)**

## **Overview**

This project generates **interactive 3D Shear Force Diagrams (SFD)** and **Bending Moment Diagrams (BMD)** for a bridge model using:

* Node coordinates
* Element connectivity
* Internal force results stored in **NetCDF** format

The diagrams are rendered using **Plotly** with hover tooltips for node IDs, coordinates, and force values.
This visualization is similar to MIDAS/SAP2000/STAAD 3D post-processing.

---

## **Features**

✅ Loads structural geometry (nodes + elements)
✅ Reads internal forces (Vy, Mz) from **screening_task.nc**
✅ Groups elements into 5 bridge girders
✅ Plots **3D SFD** (shear force extruded sideways)
✅ Plots **3D BMD** (bending moment extruded sideways)
✅ Hover labels: Node ID, X-coordinate, Vy/Mz values
✅ Separate colors for each girder
✅ Fully interactive — rotate, zoom, hover
✅ Piecewise SFD/BMD exactly as per analysis results

---

## **Directory Structure**

```
project/
│
├── node.py             # contains 'nodes' dictionary
├── element.py          # contains 'members' dictionary
├── screening_task.nc   # NetCDF internal force file
├── main.py  # your main script
└── requirements.txt #required modules to be installl
```

---

## **Input Files Explanation**

### **1. node.py**

Contains:

```python
nodes = {
    node_id: (x, y, z),
    ...
}
```

### **2. element.py**

Contains:

```python
members = {
    element_id: (node_i, node_j),
    ...
}
```

### **3. screening_task.nc**

NetCDF dataset with structure:

```
forces(Element, Component)
Component in ["Vy_i","Vy_j","Mz_i","Mz_j", ...]
```

---

## **Required Python Modules**

Install everything with:

```bash
pip install plotly xarray numpy netcdf4
```

Your script depends on:

| Module      | Purpose                            |
| ----------- | ---------------------------------- |
| **xarray**  | Reading NetCDF internal force file |
| **numpy**   | Array manipulation                 |
| **plotly**  | 3D interactive SFD/BMD plotting    |
| **netCDF4** | Backend for reading .nc files      |

---

## **How It Works**

### **1. Load Node Geometry**

* Reads node coordinates from node.py
* Reads element connectivity from element.py

### **2. Load Internal Forces**

Uses Xarray:

```python
ds = xr.open_dataset("screening_task.nc")
```

Selects components:

* **Vy_i**, **Vy_j**
* **Mz_i**, **Mz_j**

### **3. Build Polylines**

For each girder:

* Collect node coordinates
* Append force value at each node

### **4. Shift SFD/BMD**

The forces are **not** drawn on the girder itself.
They are plotted **sideways** by shifting Y-coordinate:

```python
y_plot_sfd = ys + Vy * scale
y_plot_bmd = ys + Mz * scale
```

This gives a clear 3D diagram without overlapping the structure.

### **5. Plot with Plotly**

Each girder is drawn as a separate 3D trace with hover values.

---


## 📦 **Installation**

### **1️⃣ Clone the repository**

```bash
git clone <your-repo-url>
cd project
```

### **2️⃣ Install all required Python packages**

Use the provided **requirements.txt**:

```bash
pip install -r requirements.txt
```

### **3️⃣ Run the main program**

```bash
python main.py
```



Generate:

2d plot for central grider sfd
2d bmd plot for central grider

Interactive 3D SFD

Interactive 3D BMD

Open two Plotly windows with the diagrams and also save it as html



## **Notes**

* Scaling factors for SFD (0.2) and BMD (0.05) can be adjusted.
* If results look inverted, flip the sign of Vy or Mz.
* If BMD/SFD looks too small or too large, modify scale multipliers.

---




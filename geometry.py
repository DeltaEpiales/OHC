import numpy as np
import plotly.graph_objects as go

# ==============================================================================
# OSACRA-HOLOGRAPHIC VISUALIZER: WEB LANDING PAGE (v5.0)
# ==============================================================================
# Generates 'index.html' - The main entry point for the GitHub Pages site.
# Includes interactive 3D geometry, physics layers, and navigation links.
# ==============================================================================

def generate_landing_page():
    print("--- Building Web Landing Page (index.html) ---")

    # --- 1. GEOMETRY KERNEL (Rhombic Dodecahedron) ---
    tips = np.array([
        [2.0, 0.0, 0.0], [-2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0], [0.0, -2.0, 0.0],
        [0.0, 0.0, 2.0], [0.0, 0.0, -2.0]
    ])
    corners = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]
    ])
    verts = np.vstack([tips, corners])
    
    # Triangulate Faces
    faces_quads = [
        [4, 6, 0, 8], [4, 8, 3, 12], [4, 12, 1, 10], [4, 10, 2, 6], 
        [5, 7, 0, 9], [5, 9, 3, 13], [5, 13, 1, 11], [5, 11, 2, 7], 
        [0, 6, 2, 7], [2, 10, 1, 11], [1, 12, 3, 13], [3, 8, 0, 9]  
    ]
    i, j, k = [], [], []
    for q in faces_quads:
        i.extend([q[0], q[0]]); j.extend([q[1], q[2]]); k.extend([q[2], q[3]])

    # Wireframe Edges
    edge_x, edge_y, edge_z = [], [], []
    for q in faces_quads:
        for pair in [(0,1), (1,2), (2,3), (3,0)]:
            p1, p2 = q[pair[0]], q[pair[1]]
            edge_x += [verts[p1][0], verts[p2][0], None]
            edge_y += [verts[p1][1], verts[p2][1], None]
            edge_z += [verts[p1][2], verts[p2][2], None]

    # --- 2. PHYSICS SPHERES ---
    def get_sphere(r, resolution=80):
        u, v = np.mgrid[0:2*np.pi:complex(resolution), 0:np.pi:complex(resolution)]
        x = r * np.cos(u) * np.sin(v)
        y = r * np.sin(u) * np.sin(v)
        z = r * np.cos(v)
        return x.flatten(), y.flatten(), z.flatten()

    R_out = 2.0
    R_in = 2.0 * (1.0 / np.sqrt(3)) # The Osacra Constant
    
    # --- 3. BUILD TRACES ---
    fig = go.Figure()

    # Trace 0: Lattice Skin
    fig.add_trace(go.Mesh3d(
        x=verts[:,0], y=verts[:,1], z=verts[:,2], i=i, j=j, k=k,
        color='#00FFFF', opacity=0.1, name='Lattice Geometry', hoverinfo='name'
    ))

    # Trace 1: Vertices (Nodes)
    fig.add_trace(go.Scatter3d(
        x=verts[:,0], y=verts[:,1], z=verts[:,2], mode='markers',
        marker=dict(size=4, color='white'), name='Nodes'
    ))

    # Trace 2: Edges (Links)
    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z, mode='lines',
        line=dict(color='#00FFFF', width=3), name='Links'
    ))

    # Trace 3: Bulk Sphere (Blue)
    bx, by, bz = get_sphere(R_out)
    fig.add_trace(go.Mesh3d(
        x=bx, y=by, z=bz, color='#0044FF', opacity=0.05,
        alphahull=0, name='Bulk Limit'
    ))

    # Trace 4: Horizon Sphere (Red)
    hx, hy, hz = get_sphere(R_in)
    fig.add_trace(go.Mesh3d(
        x=hx, y=hy, z=hz, color='#FF3300', opacity=0.3,
        alphahull=0, name='Event Horizon'
    ))

    # --- 4. GUI & LAYOUT ---
    updatemenus = [
        dict(
            type="buttons", direction="left", x=0.5, y=0.05, xanchor="center",
            bgcolor="#111", bordercolor="#444", borderwidth=1,
            buttons=[
                dict(label="ALL LAYERS", method="update", args=[{"visible": [True, True, True, True, True]}]),
                dict(label="GEOMETRY", method="update", args=[{"visible": [True, True, True, False, False]}]),
                dict(label="PHYSICS", method="update", args=[{"visible": [False, False, False, True, True]}]),
                dict(label="CORE", method="update", args=[{"visible": [True, True, True, False, True]}]),
            ],
            font=dict(color="cyan")
        )
    ]

    fig.update_layout(
        title={
            'text': "<b>OSACRA HOLOGRAPHIC COSMOLOGY</b><br><span style='font-size:14px;color:#888'>The Geometry of the Vacuum: α = 1/√3</span>",
            'y': 0.95, 'x': 0.5, 'xanchor': 'center'
        },
        paper_bgcolor='black', plot_bgcolor='black',
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            aspectmode='data', bgcolor='black'
        ),
        updatemenus=updatemenus,
        margin=dict(l=0, r=0, b=0, t=60),
        legend=dict(x=0.05, y=0.9, font=dict(color="#aaa"), bgcolor="rgba(0,0,0,0)")
    )
    
    # Add "View Repo" Annotation
    fig.add_annotation(
        text="<a href='https://github.com/YOUR_USERNAME/osacra-holographic-cosmology' style='color:cyan'>View Source / Paper</a>",
        xref="paper", yref="paper", x=0.98, y=0.02, showarrow=False,
        font=dict(size=12, color="cyan")
    )

    # Output to index.html with a proper tab title
    filename = 'index.html'
    fig.write_html(filename, title="Osacra Holographic Cosmology | Interactive Model")
    print(f"✅ SUCCESS: {filename} generated.")
    print("   -> Push this to GitHub and enable GitHub Pages to go live.")

if __name__ == "__main__":
    generate_landing_page()
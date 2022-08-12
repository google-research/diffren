# Diffren: a library for differentiable 3D rendering

*This is not an official Google product*

Diffren provides tools for 3D rendering with derivatives. Diffren may be used
to optimize 3D surfaces and camera parameters under pixel-based error terms,
especially as part of ML model training. The rendering pipeline is a
deferred-shading, rasterization-based pipeline. Any JAX function may be used
as a shader. Diffren provides a small number of sample shaders including
texture mapping, diffuse, and point-based lighting. A key feature of Diffren
is its implementation of
[rasterize-then-splat](https://openaccess.thecvf.com/content/ICCV2021/papers/Cole_Differentiable_Surface_Rendering_via_Non-Differentiable_Sampling_ICCV_2021_paper.pdf),
an approach for generating smooth derivatives at occlusion boundaries.

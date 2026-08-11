import sys
import os

# Add PlotNeuralNet to python path
sys.path.append(r'C:\Users\jiawe\Repos\PlotNeuralNet')
from pycore.tikzeng import *
from pycore.blocks import *

def generate_fig3a():
    tex_path = r'c:\Users\jiawe\Repos\AdaDec3D\elsarticle\figures\figure3a_architectures.tex'
    
    color_defs = r"""
\def\ConvColor{rgb:yellow,5;red,2.5;white,5}
\def\ConvReluColor{rgb:yellow,5;red,5;white,5}
\def\PoolColor{rgb:red,1;black,0.3}
\def\UnpoolColor{rgb:blue,2;green,1;black,0.3}
\def\FcColor{rgb:blue,5;red,2.5;white,5}
\def\FcReluColor{rgb:blue,5;red,5;white,4}
\def\SoftmaxColor{rgb:magenta,5;black,7}   
\def\SumColor{rgb:blue,5;green,15}
\def\edgecolor{rgb:blue,4;red,1;green,4;black,3}

\def\EncoderColor{rgb:blue,4;red,1;green,2;white,3}
\def\FullDecoderColor{rgb:blue,5;red,2;white,4}
\def\EffiDecoderColor{rgb:green,4;blue,3;white,4}
\def\DWColor{rgb:yellow,6;red,4;white,3}
\def\PWColor{rgb:blue,6;red,2;white,3}
\def\OutputColor{rgb:magenta,5;black,5}
"""

    arch = [
        to_head(r'C:/Users/jiawe/Repos/PlotNeuralNet'),
        r"\usepackage{amsmath, amsfonts, amssymb}",
        to_cor(),
        color_defs,
        to_begin(),

        # =====================================================================
        # PARADIGM 1: Full Decoder (E0)
        # =====================================================================
        r"\node[anchor=west, font=\small\bfseries\color{blue!80!black}] at (-2, 8.1, 0) {Full Baseline ($E_0$): $C_{384}$, All 4 Stages, Dense 3D Convs ($53.0$M Params, $578.7$ GMac)};",
        
        to_Conv('e0_in', '', '', offset="(-2,6.0,0)", to="(0,0,0)", width=0.8, height=6.2, depth=6.2, caption="Patch"),
        to_Conv('e0_enc1', '', '', offset="(1.4,0,0)", to="(e0_in-east)", width=1.4, height=6.2, depth=6.2, fill=r"\EncoderColor", caption="Enc1"),
        to_Conv('e0_enc2', '', '', offset="(1.2,0,0)", to="(e0_enc1-east)", width=2.0, height=4.6, depth=4.6, fill=r"\EncoderColor", caption="Enc2"),
        to_Conv('e0_enc3', '', '', offset="(1.2,0,0)", to="(e0_enc2-east)", width=2.6, height=3.2, depth=3.2, fill=r"\EncoderColor", caption="Enc3"),
        to_Conv('e0_bottleneck', '', '', offset="(1.4,0,0)", to="(e0_enc3-east)", width=3.8, height=2.0, depth=2.0, fill=r"\EncoderColor", caption="Bottleneck"),
        
        # Full Decoder Pathway
        to_Conv('e0_dec3', '', '', offset="(1.5,0,0)", to="(e0_bottleneck-east)", width=2.6, height=3.2, depth=3.2, fill=r"\FullDecoderColor", caption="Dec3"),
        to_Conv('e0_dec2', '', '', offset="(1.2,0,0)", to="(e0_dec3-east)", width=2.0, height=4.6, depth=4.6, fill=r"\FullDecoderColor", caption="Dec2"),
        to_Conv('e0_dec1', '', '', offset="(1.2,0,0)", to="(e0_dec2-east)", width=1.4, height=6.2, depth=6.2, fill=r"\FullDecoderColor", caption="Dec1"),
        to_Conv('e0_out', '', '', offset="(1.4,0,0)", to="(e0_dec1-east)", width=1.0, height=6.2, depth=6.2, fill=r"\OutputColor", caption="Mask"),
        
        # Connections & Layered Arching Skips
        to_connection('e0_in', 'e0_enc1'),
        to_connection('e0_enc1', 'e0_enc2'),
        to_connection('e0_enc2', 'e0_enc3'),
        to_connection('e0_enc3', 'e0_bottleneck'),
        to_connection('e0_bottleneck', 'e0_dec3'),
        to_connection('e0_dec3', 'e0_dec2'),
        to_connection('e0_dec2', 'e0_dec1'),
        to_connection('e0_dec1', 'e0_out'),
        to_skip('e0_enc1', 'e0_dec1', pos=1.80),
        to_skip('e0_enc2', 'e0_dec2', pos=1.50),
        to_skip('e0_enc3', 'e0_dec3', pos=1.28),

        # =====================================================================
        # PARADIGM 2: EffiDec3D (E1)
        # =====================================================================
        r"\node[anchor=west, font=\small\bfseries\color{teal!80!black}] at (-2, 3.9, 0) {EffiDec3D ($E_1$): $C_{48}$ ($-87.5\%$), Stage 1 Omitted ($3.2$M Params, $49.5$ GMac)};",
        
        to_Conv('e1_in', '', '', offset="(-2,2.2,0)", to="(0,0,0)", width=0.8, height=6.2, depth=6.2, caption="Patch"),
        to_Conv('e1_enc1', '', '', offset="(1.4,0,0)", to="(e1_in-east)", width=1.4, height=6.2, depth=6.2, fill=r"\EncoderColor", caption="Enc1"),
        to_Conv('e1_enc2', '', '', offset="(1.2,0,0)", to="(e1_enc1-east)", width=2.0, height=4.6, depth=4.6, fill=r"\EncoderColor", caption="Enc2"),
        to_Conv('e1_enc3', '', '', offset="(1.2,0,0)", to="(e1_enc2-east)", width=2.6, height=3.2, depth=3.2, fill=r"\EncoderColor", caption="Enc3"),
        to_Conv('e1_bottleneck', '', '', offset="(1.4,0,0)", to="(e1_enc3-east)", width=3.8, height=2.0, depth=2.0, fill=r"\EncoderColor", caption="Bottleneck"),
        
        # Compressed Decoder Pathway
        to_Conv('e1_dec3', '', '', offset="(1.5,0,0)", to="(e1_bottleneck-east)", width=0.6, height=3.2, depth=3.2, fill=r"\EffiDecoderColor", caption="Dec3"),
        to_Conv('e1_dec2', '', '', offset="(1.2,0,0)", to="(e1_dec3-east)", width=0.4, height=4.6, depth=4.6, fill=r"\EffiDecoderColor", caption="Dec2"),
        to_Conv('e1_out', '', '', offset="(1.4,0,0)", to="(e1_dec2-east)", width=1.0, height=6.2, depth=6.2, fill=r"\OutputColor", caption="Mask"),
        
        # Connections & Skips
        to_connection('e1_in', 'e1_enc1'),
        to_connection('e1_enc1', 'e1_enc2'),
        to_connection('e1_enc2', 'e1_enc3'),
        to_connection('e1_enc3', 'e1_bottleneck'),
        to_connection('e1_bottleneck', 'e1_dec3'),
        to_connection('e1_dec3', 'e1_dec2'),
        to_connection('e1_dec2', 'e1_out'),
        to_skip('e1_enc2', 'e1_dec2', pos=1.68),
        to_skip('e1_enc3', 'e1_dec3', pos=1.35),

        # =====================================================================
        # PARADIGM 3: Depthwise-Separable (E1')
        # =====================================================================
        r"\node[anchor=west, font=\small\bfseries\color{orange!90!black}] at (-2, -0.1, 0) {Separable ($E_1'$): $C_{384}$, All Stages, DW 3D Conv + PW 1x1x1 Conv ($37.3$M Params, $174.0$ GMac)};",
        
        to_Conv('e1s_in', '', '', offset="(-2,-2.0,0)", to="(0,0,0)", width=0.8, height=6.2, depth=6.2, caption="Patch"),
        to_Conv('e1s_enc1', '', '', offset="(1.4,0,0)", to="(e1s_in-east)", width=1.4, height=6.2, depth=6.2, fill=r"\EncoderColor", caption="Enc1"),
        to_Conv('e1s_enc2', '', '', offset="(1.2,0,0)", to="(e1s_enc1-east)", width=2.0, height=4.6, depth=4.6, fill=r"\EncoderColor", caption="Enc2"),
        to_Conv('e1s_enc3', '', '', offset="(1.2,0,0)", to="(e1s_enc2-east)", width=2.6, height=3.2, depth=3.2, fill=r"\EncoderColor", caption="Enc3"),
        to_Conv('e1s_bottleneck', '', '', offset="(1.4,0,0)", to="(e1s_enc3-east)", width=3.8, height=2.0, depth=2.0, fill=r"\EncoderColor", caption="Bottleneck"),
        
        # Factorized Decoder Pathway
        to_Conv('e1s_dec3_dw', '', '', offset="(1.5,0,0)", to="(e1s_bottleneck-east)", width=1.3, height=3.2, depth=3.2, fill=r"\DWColor", caption="DW3D"),
        to_Conv('e1s_dec3_pw', '', '', offset="(1.0,0,0)", to="(e1s_dec3_dw-east)", width=1.3, height=3.2, depth=3.2, fill=r"\PWColor", caption="PW1x1"),
        
        to_Conv('e1s_dec2_dw', '', '', offset="(1.3,0,0)", to="(e1s_dec3_pw-east)", width=1.0, height=4.6, depth=4.6, fill=r"\DWColor", caption="DW3D"),
        to_Conv('e1s_dec2_pw', '', '', offset="(1.0,0,0)", to="(e1s_dec2_dw-east)", width=1.0, height=4.6, depth=4.6, fill=r"\PWColor", caption="PW1x1"),
        
        to_Conv('e1s_dec1_dw', '', '', offset="(1.3,0,0)", to="(e1s_dec2_pw-east)", width=0.8, height=6.2, depth=6.2, fill=r"\DWColor", caption="DW3D"),
        to_Conv('e1s_dec1_pw', '', '', offset="(1.0,0,0)", to="(e1s_dec1_dw-east)", width=0.8, height=6.2, depth=6.2, fill=r"\PWColor", caption="PW1x1"),
        
        to_Conv('e1s_out', '', '', offset="(1.4,0,0)", to="(e1s_dec1_pw-east)", width=1.0, height=6.2, depth=6.2, fill=r"\OutputColor", caption="Mask"),
        
        # Connections & Skips
        to_connection('e1s_in', 'e1s_enc1'),
        to_connection('e1s_enc1', 'e1s_enc2'),
        to_connection('e1s_enc2', 'e1s_enc3'),
        to_connection('e1s_enc3', 'e1s_bottleneck'),
        to_connection('e1s_bottleneck', 'e1s_dec3_dw'),
        to_connection('e1s_dec3_dw', 'e1s_dec3_pw'),
        to_connection('e1s_dec3_pw', 'e1s_dec2_dw'),
        to_connection('e1s_dec2_dw', 'e1s_dec2_pw'),
        to_connection('e1s_dec2_pw', 'e1s_dec1_dw'),
        to_connection('e1s_dec1_dw', 'e1s_dec1_pw'),
        to_connection('e1s_dec1_pw', 'e1s_out'),
        to_skip('e1s_enc1', 'e1s_dec1_dw', pos=1.80),
        to_skip('e1s_enc2', 'e1s_dec2_dw', pos=1.50),
        to_skip('e1s_enc3', 'e1s_dec3_dw', pos=1.28),

        to_end()
    ]
    to_generate(arch, tex_path)
    print(f"Generated Figure 3a at: {tex_path}")

def generate_fig3b():
    tex_path = r'c:\Users\jiawe\Repos\AdaDec3D\elsarticle\figures\figure3b_factorial.tex'
    
    arch = [
        r"""\documentclass[border=8pt, multi, tikz]{standalone}
\usepackage{amsmath, amsfonts, amssymb}
\usetikzlibrary{positioning, arrows.meta, shapes.geometric, calc}
\begin{document}
\begin{tikzpicture}

\node[draw=purple!80!black, line width=1.2pt, fill=purple!10, rounded corners=6pt, inner sep=8pt, align=center, text width=2.6cm] (frozen_enc) at (-2.0,-3.0,0) {\textbf{\large Frozen Encoder}\\[3pt]{\small ($\mathbf{E}_{\mathrm{frozen}}$)}};

\node[font=\bfseries\large] at (8.5,-0.1,0) {Retain Stage 1 ($RF_1$)};
\node[font=\bfseries\large] at (17.5,-0.1,0) {Omit Stage 1 ($RF_2$)};

\node[font=\bfseries\small, anchor=south west, text=blue!80!black] at (1.6,-1.2,0) {Full Channel ($C_{384}$)};
\node[font=\bfseries\small, anchor=south west, text=teal!80!black] at (1.6,-4.8,0) {Reduced Channel ($C_{48}$)};

\node[draw=blue!70, fill=blue!10, line width=1.2pt, rounded corners=6pt, inner sep=6pt, text width=4.8cm, align=center] (grid11) at (8.5,-1.2,0) {\textbf{\large Full ($C_{384}, RF_1$)}\\[3pt]{\small Dice: 80.44\%}};

\node[draw=gray!70, fill=gray!10, line width=1.2pt, rounded corners=6pt, inner sep=6pt, text width=4.8cm, align=center] (grid12) at (17.5,-1.2,0) {\textbf{\large Res-Only ($C_{384}, RF_2$)}\\[3pt]{\small Dice: 77.97\%}};

\node[draw=gray!70, fill=gray!10, line width=1.2pt, rounded corners=6pt, inner sep=6pt, text width=4.8cm, align=center] (grid21) at (8.5,-4.8,0) {\textbf{\large Chan-Only ($C_{48}, RF_1$)}\\[3pt]{\small Dice: 79.80\%}};

\node[draw=teal!70, fill=teal!10, line width=1.2pt, rounded corners=6pt, inner sep=6pt, text width=4.8cm, align=center] (grid22) at (17.5,-4.8,0) {\textbf{\large Effi ($C_{48}, RF_2$)}\\[3pt]{\small Dice: 78.35\%}};

\draw[<->, draw=red, line width=1.5pt] (grid11.east) -- (grid12.west) node[midway, above=4pt, font=\bfseries\small\color{red}, align=center] {$\Delta_{RF} = -338.7$ GMac\\[2pt]($-1.96\%$ Dice)};

\draw[<->, draw=red, line width=1.5pt] (grid11.south) -- (grid21.north) node[midway, right=6pt, font=\bfseries\small\color{red}, align=center] {$\Delta_C = -44.5$M Params\\[2pt]($-0.13\%$ Dice)};

\draw[->, draw=purple!80!black, line width=1.4pt, dashed] (frozen_enc.east) -- (1.2,-3.0) |- (grid11.west);
\draw[->, draw=purple!80!black, line width=1.4pt, dashed] (frozen_enc.east) -- (1.2,-3.0) |- (grid21.west);

\end{tikzpicture}
\end{document}
"""
    ]
    with open(tex_path, "w") as f:
        f.writelines(arch)
    print(f"Generated Figure 3b at: {tex_path}")

if __name__ == '__main__':
    generate_fig3a()
    generate_fig3b()

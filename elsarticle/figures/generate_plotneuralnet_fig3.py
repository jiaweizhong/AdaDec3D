import sys
import os

# Add PlotNeuralNet to python path
sys.path.append(r'C:\Users\jiawe\Repos\PlotNeuralNet')
from pycore.tikzeng import *
from pycore.blocks import *

def generate_figure_3():
    tex_path = r'c:\Users\jiawe\Repos\AdaDec3D\elsarticle\figures\figure_design.tex'
    
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

        # Title (a)
        r"\node[anchor=north west, font=\large\bfseries] at (-2, 14, 0) {(a) Structural Decoder Intervention Paradigms (PlotNeuralNet 3D Architectures)};",

        # =====================================================================
        # PARADIGM 1: Full Decoder (E0)
        # =====================================================================
        r"\node[anchor=west, font=\bfseries\color{blue!80!black}] at (-2, 11, 0) {1. Full Baseline ($E_0$): $C_{384}$, All 4 Stages, Dense 3D Convs ($53.0$M Params, $578.7$ GMac)};",
        
        to_Conv('e0_in', '$96^3$', 1, offset="(-2,9,0)", to="(0,0,0)", width=0.8, height=8, depth=8, caption="CT Patch"),
        to_Conv('e0_enc1', '$96^3$', 48, offset="(1.2,0,0)", to="(e0_in-east)", width=1.5, height=8, depth=8, fill=r"\EncoderColor", caption="Enc1"),
        to_Conv('e0_enc2', '$48^3$', 96, offset="(1.0,0,0)", to="(e0_enc1-east)", width=2.2, height=6, depth=6, fill=r"\EncoderColor", caption="Enc2"),
        to_Conv('e0_enc3', '$24^3$', 192, offset="(1.0,0,0)", to="(e0_enc2-east)", width=3.0, height=4.5, depth=4.5, fill=r"\EncoderColor", caption="Enc3"),
        to_Conv('e0_bottleneck', '$12^3$', 384, offset="(1.0,0,0)", to="(e0_enc3-east)", width=4.5, height=3, depth=3, fill=r"\EncoderColor", caption="Bottleneck"),
        
        # Full Decoder Pathway
        to_Conv('e0_dec3', '$24^3$', 192, offset="(1.2,0,0)", to="(e0_bottleneck-east)", width=3.0, height=4.5, depth=4.5, fill=r"\FullDecoderColor", caption="Dec3"),
        to_Conv('e0_dec2', '$48^3$', 96, offset="(1.0,0,0)", to="(e0_dec3-east)", width=2.2, height=6, depth=6, fill=r"\FullDecoderColor", caption="Dec2"),
        to_Conv('e0_dec1', '$96^3$', 48, offset="(1.0,0,0)", to="(e0_dec2-east)", width=1.5, height=8, depth=8, fill=r"\FullDecoderColor", caption="Dec1"),
        to_Conv('e0_out', '$96^3$', 13, offset="(1.0,0,0)", to="(e0_dec1-east)", width=1.0, height=8, depth=8, fill=r"\OutputColor", caption="Mask"),
        
        # Connections & Skips
        to_connection('e0_in', 'e0_enc1'),
        to_connection('e0_enc1', 'e0_enc2'),
        to_connection('e0_enc2', 'e0_enc3'),
        to_connection('e0_enc3', 'e0_bottleneck'),
        to_connection('e0_bottleneck', 'e0_dec3'),
        to_connection('e0_dec3', 'e0_dec2'),
        to_connection('e0_dec2', 'e0_dec1'),
        to_connection('e0_dec1', 'e0_out'),
        to_skip('e0_enc1', 'e0_dec1', pos=1.25),
        to_skip('e0_enc2', 'e0_dec2', pos=1.25),
        to_skip('e0_enc3', 'e0_dec3', pos=1.25),

        # =====================================================================
        # PARADIGM 2: EffiDec3D (E1) - Capacity Removal
        # =====================================================================
        r"\node[anchor=west, font=\bfseries\color{teal!80!black}] at (-2, 4, 0) {2. EffiDec3D ($E_1$): $C_{48}$ ($-87.5\%$), Stage 1 Omitted ($3.2$M Params, $49.5$ GMac)};",
        
        to_Conv('e1_in', '$96^3$', 1, offset="(-2,2,0)", to="(0,0,0)", width=0.8, height=8, depth=8, caption="CT Patch"),
        to_Conv('e1_enc1', '$96^3$', 48, offset="(1.2,0,0)", to="(e1_in-east)", width=1.5, height=8, depth=8, fill=r"\EncoderColor", caption="Enc1"),
        to_Conv('e1_enc2', '$48^3$', 96, offset="(1.0,0,0)", to="(e1_enc1-east)", width=2.2, height=6, depth=6, fill=r"\EncoderColor", caption="Enc2"),
        to_Conv('e1_enc3', '$24^3$', 192, offset="(1.0,0,0)", to="(e1_enc2-east)", width=3.0, height=4.5, depth=4.5, fill=r"\EncoderColor", caption="Enc3"),
        to_Conv('e1_bottleneck', '$12^3$', 384, offset="(1.0,0,0)", to="(e1_enc3-east)", width=4.5, height=3, depth=3, fill=r"\EncoderColor", caption="Bottleneck"),
        
        # Compressed Decoder Pathway (Thinner Boxes, Stage 1 Omitted)
        to_Conv('e1_dec3', '$24^3$', 24, offset="(1.2,0,0)", to="(e1_bottleneck-east)", width=0.6, height=4.5, depth=4.5, fill=r"\EffiDecoderColor", caption="Dec3"),
        to_Conv('e1_dec2', '$48^3$', 12, offset="(1.0,0,0)", to="(e1_dec3-east)", width=0.4, height=6, depth=6, fill=r"\EffiDecoderColor", caption="Dec2"),
        to_Conv('e1_out', '$96^3$', 13, offset="(2.0,0,0)", to="(e1_dec2-east)", width=1.0, height=8, depth=8, fill=r"\OutputColor", caption="Mask"),
        
        # Connections & Skips
        to_connection('e1_in', 'e1_enc1'),
        to_connection('e1_enc1', 'e1_enc2'),
        to_connection('e1_enc2', 'e1_enc3'),
        to_connection('e1_enc3', 'e1_bottleneck'),
        to_connection('e1_bottleneck', 'e1_dec3'),
        to_connection('e1_dec3', 'e1_dec2'),
        to_connection('e1_dec2', 'e1_out'),
        to_skip('e1_enc2', 'e1_dec2', pos=1.25),
        to_skip('e1_enc3', 'e1_dec3', pos=1.25),

        # =====================================================================
        # PARADIGM 3: Depthwise-Separable (E1') - Factorization Control
        # =====================================================================
        r"\node[anchor=west, font=\bfseries\color{orange!90!black}] at (-2, -3, 0) {3. Separable ($E_1'$): $C_{384}$, All Stages, DW 3D Conv + PW 1x1x1 Conv ($37.3$M Params, $174.0$ GMac)};",
        
        to_Conv('e1s_in', '$96^3$', 1, offset="(-2,-5,0)", to="(0,0,0)", width=0.8, height=8, depth=8, caption="CT Patch"),
        to_Conv('e1s_enc1', '$96^3$', 48, offset="(1.2,0,0)", to="(e1s_in-east)", width=1.5, height=8, depth=8, fill=r"\EncoderColor", caption="Enc1"),
        to_Conv('e1s_enc2', '$48^3$', 96, offset="(1.0,0,0)", to="(e1s_enc1-east)", width=2.2, height=6, depth=6, fill=r"\EncoderColor", caption="Enc2"),
        to_Conv('e1s_enc3', '$24^3$', 192, offset="(1.0,0,0)", to="(e1s_enc2-east)", width=3.0, height=4.5, depth=4.5, fill=r"\EncoderColor", caption="Enc3"),
        to_Conv('e1s_bottleneck', '$12^3$', 384, offset="(1.0,0,0)", to="(e1s_enc3-east)", width=4.5, height=3, depth=3, fill=r"\EncoderColor", caption="Bottleneck"),
        
        # Factorized Decoder Pathway (DW Conv + PW Conv split blocks)
        to_Conv('e1s_dec3_dw', '$24^3$', 192, offset="(1.2,0,0)", to="(e1s_bottleneck-east)", width=1.5, height=4.5, depth=4.5, fill=r"\DWColor", caption="DW3D"),
        to_Conv('e1s_dec3_pw', '$24^3$', 192, offset="(0.3,0,0)", to="(e1s_dec3_dw-east)", width=1.5, height=4.5, depth=4.5, fill=r"\PWColor", caption="PW1x1"),
        
        to_Conv('e1s_dec2_dw', '$48^3$', 96, offset="(1.0,0,0)", to="(e1s_dec3_pw-east)", width=1.1, height=6, depth=6, fill=r"\DWColor", caption="DW3D"),
        to_Conv('e1s_dec2_pw', '$48^3$', 96, offset="(0.3,0,0)", to="(e1s_dec2_dw-east)", width=1.1, height=6, depth=6, fill=r"\PWColor", caption="PW1x1"),
        
        to_Conv('e1s_dec1_dw', '$96^3$', 48, offset="(1.0,0,0)", to="(e1s_dec2_pw-east)", width=0.8, height=8, depth=8, fill=r"\DWColor", caption="DW3D"),
        to_Conv('e1s_dec1_pw', '$96^3$', 48, offset="(0.3,0,0)", to="(e1s_dec1_dw-east)", width=0.8, height=8, depth=8, fill=r"\PWColor", caption="PW1x1"),
        
        to_Conv('e1s_out', '$96^3$', 13, offset="(1.0,0,0)", to="(e1s_dec1_pw-east)", width=1.0, height=8, depth=8, fill=r"\OutputColor", caption="Mask"),
        
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
        to_skip('e1s_enc1', 'e1s_dec1_dw', pos=1.25),
        to_skip('e1s_enc2', 'e1s_dec2_dw', pos=1.25),
        to_skip('e1s_enc3', 'e1s_dec3_dw', pos=1.25),

        # =====================================================================
        # PARADIGM (B): 2x2 Causal Factorial Grid (Frozen Shared Encoder)
        # =====================================================================
        r"\node[anchor=north west, font=\large\bfseries] at (-2, -10, 0) {(b) Causal $2 \times 2$ Factorial Matrix (Decoder Factor Attribution under Frozen Shared Encoder $\mathbf{E}_{\mathrm{frozen}}$)};",
        
        r"""
\node[draw=purple!80!black, line width=1pt, fill=purple!10, rounded corners=4pt, inner sep=6pt, align=center, text width=3.2cm] (frozen_enc) at (-2.0,-14.2,0) {\textbf{Frozen Encoder $\mathbf{E}_{\mathrm{frozen}}$}\\[3pt]{\small (Shared 3D UX-Net encoder, weights frozen)}};

\node[font=\bfseries] at (6.5,-10.8,0) {Retain Stage 1 ($RF_1$)};
\node[font=\bfseries] at (13.5,-10.8,0) {Omit Stage 1 ($RF_2$)};

\node[font=\bfseries, rotate=90] at (3.2,-12.5,0) {Full Channel ($C_{384}$)};
\node[font=\bfseries, rotate=90] at (3.2,-15.8,0) {Reduced Channel ($C_{48}$)};

\node[draw=blue!70, fill=blue!10, line width=1pt, rounded corners=4pt, inner sep=6pt, text width=3.8cm, align=center] (grid11) at (6.5,-12.5,0) {\textbf{Full ($C_{384}, RF_1$)}\\{\small 46.7M P $\cdot$ 657.8 GMac}\\[2pt]{\bfseries Dice: 80.44\%}};
\node[draw=gray!70, fill=gray!10, line width=1pt, rounded corners=4pt, inner sep=6pt, text width=3.8cm, align=center] (grid12) at (13.5,-12.5,0) {\textbf{Res-Only ($C_{384}, RF_2$)}\\{\small 46.3M P $\cdot$ 319.1 GMac}\\[2pt]{\bfseries Dice: 77.97\%}};
\node[draw=gray!70, fill=gray!10, line width=1pt, rounded corners=4pt, inner sep=6pt, text width=3.8cm, align=center] (grid21) at (6.5,-15.8,0) {\textbf{Chan-Only ($C_{48}, RF_1$)}\\{\small 2.2M P $\cdot$ 388.2 GMac}\\[2pt]{\bfseries Dice: 79.80\%}};
\node[draw=teal!70, fill=teal!10, line width=1pt, rounded corners=4pt, inner sep=6pt, text width=3.8cm, align=center] (grid22) at (13.5,-15.8,0) {\textbf{Effi ($C_{48}, RF_2$)}\\{\small 1.8M P $\cdot$ 49.5 GMac}\\[2pt]{\bfseries Dice: 78.35\%}};

\draw[<->, draw=red, line width=1.3pt] (grid11.north east) -- (grid12.north west) node[midway, above=4pt, font=\small\bfseries\color{red}] {$\Delta_{RF} = -338.7$ GMac ($-1.96\%$ Dice)};
\draw[<->, draw=red, line width=1.3pt] (grid11.south) -- (grid21.north) node[midway, right=6pt, font=\small\bfseries\color{red}] {$\Delta_C = -44.5$M Params ($-0.13\%$ Dice)};
\draw[->, draw=purple!80!black, line width=1.2pt, dashed] (frozen_enc.east) -- (grid11.west);
\draw[->, draw=purple!80!black, line width=1.2pt, dashed] (frozen_enc.east) -- (grid21.west);
""",

        to_end()
    ]
    
    to_generate(arch, tex_path)
    print(f"Successfully generated clean PlotNeuralNet TikZ at: {tex_path}")

if __name__ == '__main__':
    generate_figure_3()

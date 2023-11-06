%% create a mask in CleWin5

%% Useful commands
% rectangle(x1, y1, x2, y2)
% box(x, y, width, height, angle)
% polygon(nodes)
% wire(style, width, nodes)
% circle(x, y, radius)
% ring(x, y, radius, width)
% text(text, matrix)
% symbol(symbolname, matrix)

%% Zone Rectangle Drawer

L1_full = 101600; % L stands for length
M1 = 4000; % M stands for margin
L1 = L1_full - 2*M1;
O1_full = - L1_full/2; % O stand for origin
O1 = O1_full + M1;
R1 = 3; % C stands for rows
C1 = 3; % C stands for columns

L2_full = L1 / 3;
M2 = 2000;
L2 = L2_full - 2*M2;


for x1=0:R1-1
    for y1=0:C1-1
        O2x = O1 + x1*L2_full + M2;
        O2y = O1 + y1*L2_full + M2;
        rectangle(O2x, O2y, O2x+L2, O2y+L2);       
    end
end


%% 20um

% SPACE = 100
% L1_full = 101600; % L stands for length
% M1 = 4000; % M stands for margin
% L1 = L1_full - 2*M1;
% O1_full = - L1_full/2; % O stand for origin
% O1 = O1_full + M1;
% 
% N2 = 3; % N stands for number of rows/columns
% L2_full = L1 / N2;
% M2 = 2000;
% L2 = L2_full - 2*M2;
% 
% N3 = 20;
% L3_full = L2 / N3;
% M3 = 180;
% L3 = L3_full - 2*M3;
% 
% N4 = 10;
% space = L3/N4;
% diameter = 20;
% radius = diameter/2;


% SPACE = 80
% L1_full = 101600; % L stands for length
% M1 = 4000; % M stands for margin
% L1 = L1_full - 2*M1;
% O1_full = - L1_full/2; % O stand for origin
% O1 = O1_full + M1;
% 
% N2 = 3; % N stands for number of rows/columns
% L2_full = L1 / N2;
% M2 = 2400;
% L2 = L2_full - 2*M2;
% 
% N3 = 15;
% L3_full = L2 / N3;
% M3 = 80;
% L3 = L3_full - 2*M3;
% 
% N4 = 20;
% space = L3/N4;
% diameter = 20;
% radius = diameter/2;


% SPACE = 70
L1_full = 101600; % L stands for length
M1 = 4600; % M stands for margin
L1 = L1_full - 2*M1;
O1_full = - L1_full/2; % O stand for origin
O1 = O1_full + M1;

N2 = 3; % N stands for number of rows/columns
L2_full = L1 / N2;
M2 = 3000;
L2 = L2_full - 2*M2;

N3 = 16;
L3_full = L2 / N3;
M3 = 75;
L3 = L3_full - 2*M3;

N4 = 20;
space = L3/N4;
diameter = 20;
radius = diameter/2;

for x2=0:N2-1
    for y2=0:N2-1
        O2x = O1 + x2*L2_full + M2;
        O2y = O1 + y2*L2_full + M2;
        
        for x3=0:N3-1
            for y3=0:N3-1
                O3x = O2x + x3*L3_full + M3;
                O3y = O2y + y3*L3_full + M3;
                
                for x4=0:N4-1
                    for y4=0:N4-1
                        X = O3x + x4*space;
                        Y = O3y + y4*space;
                        circle(X, Y, radius);
                    end
                end
                
            end
        end
        
    end
end

% Informations

Ntot = N3*N3 * N4*N4;
Stot = L2*1e-6 * L2*1e-6;
S_cs20 = pi * (10*1e-3) * (10*1e-3);
Ratio_S = S_cs20/Stot;
Expected_Ntot = Ntot*Ratio_S;



%% 20um V2

% SPACE = 100
% L1_full = 101600; % L stands for length
% M1 = 4000; % M stands for margin
% L1 = L1_full - 2*M1;
% O1_full = - L1_full/2; % O stand for origin
% O1 = O1_full + M1;
% 
% N2 = 3; % N stands for number of rows/columns
% L2_full = L1 / N2;
% M2 = 2000;
% L2 = L2_full - 2*M2;
% 
% N3 = 20;
% L3_full = L2 / N3;
% M3 = 180;
% L3 = L3_full - 2*M3;
% 
% N4 = 10;
% space = L3/N4;
% diameter = 20;
% radius = diameter/2;


% SPACE = 80
% L1_full = 101600; % L stands for length
% M1 = 4000; % M stands for margin
% L1 = L1_full - 2*M1;
% O1_full = - L1_full/2; % O stand for origin
% O1 = O1_full + M1;
% 
% N2 = 3; % N stands for number of rows/columns
% L2_full = L1 / N2;
% M2 = 2400;
% L2 = L2_full - 2*M2;
% 
% N3 = 15;
% L3_full = L2 / N3;
% M3 = 80;
% L3 = L3_full - 2*M3;
% 
% N4 = 20;
% space = L3/N4;
% diameter = 20;
% radius = diameter/2;


% SPACE = 70
L1_full = 101600; % L stands for length
M1 = 4000; % M stands for margin
L1 = L1_full - 2*M1;
O1_full = - L1_full/2; % O stand for origin
O1 = O1_full + M1;

N3 = 60;
L3_full = L1 / N3;
M3 = 80;
L3 = L3_full - 2*M3;

N4 = 20;
space = L3/N4;
diameter = 20;
radius = diameter/2;

        
for x3=0:N3-1
    for y3=0:N3-1
        O3x = O1 + x3*L3_full + M3;
        O3y = O1 + y3*L3_full + M3;

        for x4=0:N4-1
            for y4=0:N4-1
                X = O3x + x4*space;
                Y = O3y + y4*space;
                circle(X, Y, radius);
            end
        end

    end
end

% Informations

Ntot = N3*N3 * N4*N4;
Stot = L1*1e-6 * L1*1e-6;
C = Ntot/Stot;
S_cs20 = pi * (10*1e-3) * (10*1e-3);
S_cs25 = pi * (12.5*1e-3) * (12.5*1e-3);
Expected_Ntot = C*S_cs25;


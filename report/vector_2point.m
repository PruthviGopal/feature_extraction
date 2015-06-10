% 3D mesh plot 
figure(1); 


%surf(xx,yy,G); 
x = [1, 2];
y = [1, 2];
z = [1, 2];
scatter3(x, y, z);

hold on;

%colormap('jet'); 
%shading interp; 
plot3(x, y, z);
text(x(1),y(1),z(1), 'x1, y1, z1');
text(x(2),y(2),z(2), 'x2, y2, z2');
title('length^{2} = (x1 - x2)^{2} + (y1 - y2)^{2} + (z1 - z2)^{2}');
hold on;
axis([0 2 0 2 0 2.5]);


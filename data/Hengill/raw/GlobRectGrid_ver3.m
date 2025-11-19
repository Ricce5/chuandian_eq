function [LonGMat,LatGMat,areamat,grid_vec_format] = GlobRectGrid_ver3(resol,lonbdn,latbnd)


longrid=[lonbdn(1):resol(1):lonbdn(2)]';
longridcen = longrid(1:end-1)/2+longrid(2:end)/2; % Center of the longitude grid cells

latgrid = [latbnd(1):resol(2):latbnd(2)]';
latgridcen=latgrid(1:end-1)/2+latgrid(2:end)/2; % Center of the latitude grid cells

[LonGMat,LatGMat]=meshgrid(longridcen,latgridcen);
LonG=LonGMat(:); % transform to vector
LatG=LatGMat(:);
E=referenceEllipsoid('wgs84','kilometers');

area = areaquad(LatG-resol(2)/2,LonG-resol(1)/2,LatG+resol(2)/2,LonG+resol(1)/2,E); % calculate area of each cell in km^2

areamat=vec2mat(area,size(LonGMat,1))';
check=1;
grid_vec_format = [LonGMat(:),LatGMat(:),areamat(:)];
end

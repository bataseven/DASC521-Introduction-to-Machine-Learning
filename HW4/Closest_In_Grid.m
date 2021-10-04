A = rand(2000,2); % point positions
[X, Y] = meshgrid(linspace(0,0.1,5),linspace(0,0.1,5));
B = [X(:), Y(:)]; % grid positions
% crop B to area A
dxA = max(A(:,1))-min(A(:,1));
dyA = max(A(:,2))-min(A(:,2));
scaleWind = 0.1;
xRangeA = [min(A(:,1))-dxA*scaleWind max(A(:,1))+dxA*scaleWind];
yRangeA = [min(A(:,2))-dyA*scaleWind max(A(:,2))+dyA*scaleWind];
idxB_inA = xRangeA(1)<=B(:,1) & B(:,1)<=xRangeA(2) & ...
           yRangeA(1)<=B(:,2) & B(:,2)<=yRangeA(2);
B = B(idxB_inA,:);
% coase search: find points in squares centered at grid points
Npoints = size(A,1);               % number of points
Ngrid   = size(B,1);               % number of points on grid
dX = max(B(:,1))-min(B(:,1));
dY = max(B(:,2))-min(B(:,2));
% charactersitic length for the square (empirical determined)
dL = max(abs([dX;dY]))/sqrt((Npoints+Ngrid)/20); 
idxPreSearch =  A(:,1)>B(:,1).'-dL & A(:,1)<B(:,1).'+dL & ...
                A(:,2)>B(:,2).'-dL & A(:,2)<B(:,2).'+dL;
% find index-number of closest point for each grid-item
nIdxPointMinAll = nan(Ngrid,1);
for iGrid=1:Ngrid
    idxTemp = idxPreSearch(:,iGrid);
    nIdxTemp = find(idxTemp);
    % compute Euclidean distances
    dist2 = sum((B(iGrid,:) - A(idxTemp,:)).^ 2, 2);
    nIdexPointMin = dist2 == min(dist2); 
    if ~isempty(nIdexPointMin)
        nIdxPointMinAll(iGrid) = nIdxTemp(nIdexPointMin);
    end
end
nIdxPointMinAll(isnan(nIdxPointMinAll)) = [];
nIdxPointMinAll = unique(nIdxPointMinAll);
figure; hold on; plot(A(:,1),A(:,2),'.g'); 
plot(B(:,1),B(:,2),'*r')
plot(A(idxPreSearch(:,floor(Ngrid/3)),1),A(idxPreSearch(:,floor(Ngrid/3)),2),'.k')
plot(A(nIdxPointMinAll,1),A(nIdxPointMinAll,2),'+b')
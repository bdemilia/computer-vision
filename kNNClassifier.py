Pkg.add("Images")
Pkg.add("DataFrames")

using Images
using DataFrames

#typeData could be either "train" or "test.
#labelsInfo should contain the IDs of each image to be read
#The images in the trainResized and testResized data files
#are 20x20 pixels, so imageSize is set to 400.
#path should be set to the location of the data files
function read_data(typeData, labelsInfo, imageSize, path)
 #Intialize x matrix
 x = zeros(size(labelsInfo, 1), imageSize)

 for (index, idImage) in enumerate(labelsInfo[:ID])
  #Read image file
  nameFile = "$(path)/$(typeData)Resized/$(idImage).Bmp"
  img = load(nameFile)
  #Convert img to float values
  temp = convert(Image{Images.Gray}, img)

  #Convert color images to gray images
  #by taking the average of the color scales.
  if ndims(temp) == 3
   temp = mean(temp.data, 1)
  end

  #Transform image matrix to a vector and store
  #it in data matrix
  x[index, :] = reshape(temp, 1, imageSize)
 end
 return x
end

#############################################

imageSize = 400 # 20 x 20 pixel

#Set location of data files, folders
path = "C:\\Users\\Ben\\z-streetview-julia\\"

#Read information about training data , IDs.
labelsInfoTrain = readtable("$(path)/trainLabels.csv")
println("10%")

#Read training matrix
xTrain = read_data("train", labelsInfoTrain, imageSize, path)
println("20%")

#Read information about test data ( IDs ).
labelsInfoTest = readtable("$(path)/sampleSubmission.csv")
println("30%")

#Read test matrix
xTest = read_data("test", labelsInfoTest, imageSize, path)
println("40%")

#Get only first character of string (convert from string to character).
#Apply the function to each element of the column "Class"
yTrain = map(x -> x[1], labelsInfoTrain[:Class])
println("50%")

#Convert from character to integer
yTrain = convert(Array{Int32}, yTrain)
println("60%")

########################################

@everywhere function euclidean_distance(a, b)
 distance = 0.0 
 for index in 1:size(a, 1) 
  distance += (a[index]-b[index]) * (a[index]-b[index])
 end
 return distance
end

@everywhere function get_k_nearest_neighbors(xTrain, imageI, k)
 nRows, nCols = size(xTrain) 
 imageJ = Array(Float32, nRows)
 distances = Array(Float32, nCols) 
 for j in 1:nCols
  for index in 1:nRows
   imageJ[index] = xTrain[index, j]
  end
  distances[j] = euclidean_distance(imageI, imageJ)
 end
 sortedNeighbors = sortperm(distances)
 kNearestNeighbors = sortedNeighbors[1:k]
 return kNearestNeighbors
end 

@everywhere function assign_label(xTrain, yTrain, k, imageI)
 kNearestNeighbors = get_k_nearest_neighbors(xTrain, imageI, k) 
 counts = Dict{Int, Int}() 
 highestCount = 0
 mostPopularLabel = 0
 for n in kNearestNeighbors
  labelOfN = yTrain[n]
  if !haskey(counts, labelOfN)
   counts[labelOfN] = 0
  end
  counts[labelOfN] += 1 #add one to the count
  if counts[labelOfN] > highestCount
   highestCount = counts[labelOfN]
   mostPopularLabel = labelOfN
  end 
 end
 return mostPopularLabel
end

k = 3 # The CV accuracy shows this value to be the best.
yPredictions = @parallel (vcat) for i in 1:size(xTest, 2)
 nRows = size(xTrain, 1)
 imageI = Array(Float32, nRows)
 for index in 1:nRows
  imageI[index] = xTest[index, i]
 end
 assign_label(xTrain, yTrain, k, imageI)
end

#Convert integer predictions to character
labelsInfoTest["Class"] = char(yPredictions)

#Save predictions
writetable("$(path)/juliaKNNSubmission.csv", labelsInfoTest, separator=',', header=true)

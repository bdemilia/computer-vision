using Images
using DataFrames

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

####################################

imageSize = 400 # 20 x 20 pixel

#Set location of data files, folders
path = ""

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

####################################

using DecisionTree

#Train random forest with
#20 for number of features chosen at each random split,
#50 for number of trees,
#and 1.0 for ratio of subsampling.
model = build_forest(yTrain, xTrain, 20, 50, 1.0)
println("70%")

#Get predictions for test data
predTest = apply_forest(model, xTest)
println("80%")

#Convert integer predictions to character
labelsInfoTest[:Class] = map(Char, predTest)
println("90%")



writetable("$(path)/juliaSubmission.csv", labelsInfoTest, separator=',', header=true)
println("100%")

accuracy = nfoldCV_forest(yTrain, xTrain, 20, 50, 4, 1.0);

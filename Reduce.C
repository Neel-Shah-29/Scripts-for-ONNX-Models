#include "TMVA/SOFIE_common.hxx"

int Reduce(const float * input, float * output, const std::vector<size_t> & inputShape, int axis) {

   std::vector<size_t> outputShape;

   outputShape = inputShape;
   outputShape[axis] = 1;

   size_t outputLength = TMVA::Experimental::SOFIE::ConvertShapeToLength(outputShape);

   auto inputStrides = TMVA::Experimental::SOFIE::UTILITY::ComputeStrideFromShape(inputShape);
   auto outputStrides = TMVA::Experimental::SOFIE::UTILITY::ComputeStrideFromShape(outputShape);

   size_t dim = outputShape.size();

   std::vector<size_t> idx(dim);


   for (size_t i = 0; i < outputLength; i++) {
      
   // write here according to size of shape
   // in generation code can be done automatically
   // i0 =  i / s0 ; i1 = (i % s0) / s1 ; i2 = ( (i % s0) % s1 ) / s2 and so on
   // and we have for the inverse
   // i = i0 * s0 + i1 * s1 + i2 * s2 + i3 * s3 ....

      // don't need to divide by last stride s[n-1] since it is 1 by definition
      if (dim == 2) {
         idx[0] = i / outputStrides[0];
         idx[1] = i % outputStrides[0];
      }
      if (dim == 3) {
         idx[0] = i / outputStrides[0];
         idx[1] = (i % outputStrides[0]) / outputStrides[1];
         idx[2] = (i % outputStrides[0]) % outputStrides[1];
      }
      if (dim == 4) {
         idx[0] = i / outputStrides[0];
         idx[1] = (i % outputStrides[0]) / outputStrides[1];
         idx[2] = ((i % outputStrides[0]) % outputStrides[1]) / outputStrides[2]; 
         idx[3] = ((i % outputStrides[0]) % outputStrides[1]) % outputStrides[2];
      }

      assert(idx[axis] == 0);  // we can avoid computing this for the reduction axis which by definition is always zero 


      // input index
      size_t j = 0;
      if (dim == 2) j = idx[0]*inputStrides[0] + idx[1];
      if (dim == 3) j = idx[0]*inputStrides[0] + idx[1]* inputStrides[1] + idx[2];
      if (dim == 4) j = idx[0]*inputStrides[0] + idx[1]* inputStrides[1] + idx[2]*inputStrides[2] + idx[3];
      
      // now we compute the reduction
      // e.g. sum and then mean
      float sum = 0;
      for (size_t k = 0; k < inputShape[axis]; k++) {
         idx[axis] = k;
         // compute input index j 
         size_t j = 0;
         if (dim == 2) j = idx[0]*inputStrides[0] + idx[1];
         if (dim == 3) j = idx[0]*inputStrides[0] + idx[1]* inputStrides[1] + idx[2];
         if (dim == 4) j = idx[0]*inputStrides[0] + idx[1]* inputStrides[1] + idx[2]*inputStrides[2] + idx[3];

         sum += input[j];
      }
      float average = sum/float(inputShape[axis]);
      output[i] = average;

   }
   return outputLength;
}


void testReduce(int axis = 0) {

   std::vector<float> x = { 1,2,3,4,5,6,7,8,9,10,11,12 };
   std::vector<size_t> shape = {2,3, 2};
   

   //int axis = 0;

   std::vector<float> y(x.size());

   auto n = Reduce(x.data(), y.data(), shape, axis);

   y.resize(n);

   for (int i = 0; i < n; i++)
      std::cout << y[i] << " , ";

   std::cout << std::endl;
   

}


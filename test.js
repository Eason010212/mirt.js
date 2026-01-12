import MIRT from './mirt.js';
// 1. Setup Data (5 People, 3 Items)
const responseData = [
    [1, 0, 1],
    [1, 1, 1],
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 1]
  ];
  
  const mirt = new MIRT(1); // 1-dimensional model
  
  // 2. Fit the model (2PL)
  mirt.fit(responseData, { modelType: '2PL', maxIter: 20 }).then(items => {
    console.log("Fitted Items:", items);
  
    // 3. Score a new person
    const newPerson = [1, 1, 0];
    const theta = mirt.scoreEAP(newPerson);
    console.log("Estimated Theta:", theta);
  });
/**
 *  @file   larpandoracontent/LArWorkshop/CerberusAlgorithm.cc
 *
 *  @brief  Implementation of the Cerberus algorithm class.
 *
 *  $Log: $ 
 */

#include "Pandora/AlgorithmHeaders.h"
#include "larpandoracontent/LArDeepLearning/CerberusAlgorithm.h"
#include "larpandoracontent/LArHelpers/LArPfoHelper.h"
#include "larpandoracontent/LArHelpers/LArGeometryHelper.h"
#include "larpandoracontent/LArHelpers/LArClusterHelper.h"

#include <cmath>

using namespace pandora;
using namespace torch::indexing;


namespace lar_content{

	StatusCode CerberusAlgorithm::Run()
	{
		// ###### Get CaloHits ######
		const CaloHitList *pCaloHitListU(nullptr);
		const CaloHitList *pCaloHitListV(nullptr);
		const CaloHitList *pCaloHitListW(nullptr);		
		PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::GetList(*this, m_caloHitListNames[0], pCaloHitListU));
		PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::GetList(*this, m_caloHitListNames[1], pCaloHitListV));
		PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::GetList(*this, m_caloHitListNames[2], pCaloHitListW));
		CaloHitVector caloHitVectorU(pCaloHitListU->begin(), pCaloHitListU->end());
		CaloHitVector caloHitVectorV(pCaloHitListV->begin(), pCaloHitListV->end());
		CaloHitVector caloHitVectorW(pCaloHitListW->begin(), pCaloHitListW->end());

		const PfoList *pPfoList(nullptr);
		//PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::GetList(*this, m_pfoListNames[0], pPfoList));
		PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::GetCurrentList(*this, pPfoList));

		bool foundSuitableShower(false);
		CartesianVector vert(0.f,0.f,0.f);

		for (const ParticleFlowObject *const pPfo : *pPfoList) // Finds and adds showers to pfoListCrop
		{
			//std::cout<<" LArPfoHelper::IsShower(pPfo): "<<LArPfoHelper::IsShower(pPfo)<<"   LArPfoHelper::IsNeutrinoFinalState(pPfo): "<<LArPfoHelper::IsNeutrinoFinalState(pPfo)<<std::endl;
			if (LArPfoHelper::IsShower(pPfo) && LArPfoHelper::IsNeutrinoFinalState(pPfo)) 
			{	
				unsigned int totalHits(0);
			    ClusterList clusterList;
			    LArPfoHelper::GetTwoDClusterList(pPfo, clusterList);
			    for (const Cluster *const pCluster : clusterList)
			    {
		        	totalHits += pCluster->GetNCaloHits();
			    }
				
				std::cout<<"totalHits: "<<totalHits<<std::endl;
				if(totalHits>5)
				{
					foundSuitableShower=true;
					vert =  LArPfoHelper::GetVertex(pPfo)->GetPosition();
					break;
				}
			}
		}


		if(!foundSuitableShower) return STATUS_CODE_SUCCESS; // Skipps further processing of events with no suitable shower 

		float minX(0);
		float minZ_U(0), minZ_V(0), minZ_W(0);

		///////////////////////////////////////////////////////////////////////////////////////
		/// Find common minX
		const CartesianVector vertU = LArGeometryHelper::ProjectPosition(this->GetPandora(), vert, TPC_VIEW_U); // Project 3D vertex onto 2D U view
		const CartesianVector vertV = LArGeometryHelper::ProjectPosition(this->GetPandora(), vert, TPC_VIEW_V); // Project 3D vertex onto 2D V view
		const CartesianVector vertW = LArGeometryHelper::ProjectPosition(this->GetPandora(), vert, TPC_VIEW_W); // Project 3D vertex onto 2D W view
		
	    std::array<float, SEG>  hitDensity= {0}; // Always combining 8 wires
	    FillMinimizationArray(hitDensity, pPfoList, pCaloHitListU, vertU, vertU.GetX(), vertU.GetZ()-IMSIZE/3*0.3, true, TPC_VIEW_U); // vertU.GetX() == vertV.GetX() == vertW.GetX()
	    FillMinimizationArray(hitDensity, pPfoList, pCaloHitListV, vertV, vertV.GetX(), vertV.GetZ()-IMSIZE/3*0.3, true, TPC_VIEW_V);
	    FillMinimizationArray(hitDensity, pPfoList, pCaloHitListW, vertW, vertW.GetX(), vertW.GetZ()-IMSIZE/3*0.3, true, TPC_VIEW_W);
	    minX = FindMin(hitDensity, vertU.GetX());

	    if(minX > vertU.GetX()-10/0.3) minX = vertU.GetX()-10/0.3;
	    else if(minX < vertU.GetX()-IMSIZE*0.3+10/0.3) minX = vertU.GetX()-IMSIZE*0.3+10/0.3;

		///////////////////////////////////////////////////////////////////////////////////////
		/// Find minZ in U-view
		hitDensity= {0}; // Always combining 8 wires
		FillMinimizationArray(hitDensity, pPfoList, pCaloHitListU, vertU, vertU.GetZ(), minX, false, TPC_VIEW_U);
		minZ_U = FindMin(hitDensity, vertU.GetZ());

	    if(minZ_U > vertU.GetZ()-10/0.3) minZ_U = vertU.GetZ()-10/0.3;
	    else if(minZ_U < vertU.GetZ()-IMSIZE*0.3+10*0.3) minZ_U = vertU.GetZ()-IMSIZE*0.3+10/0.3;

		///////////////////////////////////////////////////////////////////////////////////////
		/// Find minZ in V-view
		hitDensity= {0}; // Always combining 8 wires
		FillMinimizationArray(hitDensity, pPfoList, pCaloHitListV, vertV, vertV.GetZ(), minX, false, TPC_VIEW_V);
		minZ_V = FindMin(hitDensity, vertV.GetZ());

	    if(minZ_V > vertV.GetZ()-10/0.3) minZ_V = vertV.GetZ()-10/0.3;
	    else if(minZ_V < vertV.GetZ()-IMSIZE*0.3+10*0.3) minZ_V = vertV.GetZ()-IMSIZE*0.3+10/0.3;

		///////////////////////////////////////////////////////////////////////////////////////
		/// Find minZ in W-view
		hitDensity= {0}; // Always combining 8 wires
		FillMinimizationArray(hitDensity, pPfoList, pCaloHitListW, vertW, vertW.GetZ(), minX, false, TPC_VIEW_W);
		minZ_W = FindMin(hitDensity, vertW.GetZ());

	    if(minZ_W > vertW.GetZ()-10/0.3) minZ_W = vertW.GetZ()-10/0.3;
	    else if(minZ_W < vertW.GetZ()-IMSIZE*0.3+10*0.3) minZ_W = vertW.GetZ()-IMSIZE*0.3+10/0.3;


  		torch::NoGradGuard guard;
		///////////////////////////////////////////////////////////////////////////////////////
		/// Populate input tensor to Cerberus network
		torch::Tensor tensor = torch::zeros({1,6,IMSIZE,IMSIZE}, torch::kFloat32); //Creates the data tensor
		PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, WriteDetectorGaps(tensor, minZ_U, minZ_V, minZ_W));
		PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PopulateImage(tensor, caloHitVectorU, 0, minX, minZ_U));
		PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PopulateImage(tensor, caloHitVectorV, 1, minX, minZ_V));
		PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PopulateImage(tensor, caloHitVectorW, 2, minX, minZ_W));

		// std::cout<<"CerberusPoint 7"<<std::endl;
		// ###### Load Torch model ######
  		
		torch::jit::script::Module module;
		// std::cout<<"CerberusPoint 7.1"<<std::endl;
		try {
			// std::cout<<"CerberusPoint 7.2"<<std::endl;
			// Deserialize the ScriptModule from a file using torch::jit::load().
			module = torch::jit::load("/uboone/app/users/jdetje/PanLee_v08_57_00/dev/srcs/larpandoracontent/larpandoracontent/LArDeepLearning/traced_resnet_model_CerberusU2_Jul10.pt");
			// std::cout<<"CerberusPoint 7.3"<<std::endl;
		}
		catch (const c10::Error& e) {
			std::cout << "CerberusAlgorithm::Run() - Could not load Torch model"<<std::endl;
			return STATUS_CODE_FAILURE;
		}


		// ############## Testing
		std::ofstream file("/uboone/app/users/jdetje/PanLee_v08_57_00/DeepTesting/pos.bin", std::ios::out | std::ios::binary); 
		std::array<int, 8> pos = {0};
		pos[0] = (int) ((vertU.GetX() - minX)/0.3f);
		pos[1] = (int) ((vertU.GetZ() - minZ_U)/0.3f);
		pos[2] = (int) ((vertV.GetZ() - minZ_V)/0.3f);
		pos[3] = (int) ((vertW.GetZ() - minZ_W)/0.3f);
		
		pos[4] = (int) minX;
		pos[5] = (int) minZ_U;
		pos[6] = (int) minZ_V;
		pos[7] = (int) minZ_W;
		file.write((char*)&pos, sizeof(pos));
		file.close();

		torch::Tensor pandoraReco = torch::zeros({1,3,IMSIZE,IMSIZE}, torch::kFloat32); //Creates the data tensor
		PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PopulatePandoraReconstructionTensor(pandoraReco, pPfoList, TPC_VIEW_U, 0, minX, minZ_U));
		PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PopulatePandoraReconstructionTensor(pandoraReco, pPfoList, TPC_VIEW_V, 1, minX, minZ_V));
		PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PopulatePandoraReconstructionTensor(pandoraReco, pPfoList, TPC_VIEW_W, 2, minX, minZ_W));
		torch::save(pandoraReco, torch::str("/uboone/app/users/jdetje/PanLee_v08_57_00/DeepTesting/CerberusPandoraReco.pt")); //Test_jdetje/
		
		torch::Tensor availability = torch::zeros({1,3,IMSIZE,IMSIZE}, torch::kFloat32); //Creates the data tensor
		PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PopulateAvailabilityTensor(availability, caloHitVectorU, 0, minX, minZ_U));
		PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PopulateAvailabilityTensor(availability, caloHitVectorV, 1, minX, minZ_V));
		PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PopulateAvailabilityTensor(availability, caloHitVectorW, 2, minX, minZ_W));

		torch::Tensor mctruth = torch::zeros({1,3,IMSIZE,IMSIZE}, torch::kFloat32); //Creates the data tensor
		PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PopulateMCTensor(mctruth, caloHitVectorU, 0, minX, minZ_U));
		PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PopulateMCTensor(mctruth, caloHitVectorV, 1, minX, minZ_V));
		PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PopulateMCTensor(mctruth, caloHitVectorW, 2, minX, minZ_W));
		torch::save(mctruth, torch::str("/uboone/app/users/jdetje/PanLee_v08_57_00/DeepTesting/CerberusMC.pt"));
		// ############## Testing End


		torch::save(tensor, torch::str("/uboone/app/users/jdetje/PanLee_v08_57_00/DeepTesting/CerberusInput.pt"));
		std::cout<<"CerberusPoint 8"<<std::endl;
		std::vector<torch::jit::IValue> inputs;
		inputs.push_back(tensor);
		std::cout<<"CerberusPoint 8.01"<<std::endl;
		at::Tensor output = module.forward(inputs).toTensor();
		std::cout<<"CerberusPoint 8.02"<<std::endl;
		torch::save(output, torch::str("/uboone/app/users/jdetje/PanLee_v08_57_00/DeepTesting/CerberusOutput.pt"));
		at::Tensor outputU = output.index({Slice(), Slice(0,3), Slice(), Slice()}).argmax(1);
		at::Tensor outputV = output.index({Slice(), Slice(3,6), Slice(), Slice()}).argmax(1);
		at::Tensor outputW = output.index({Slice(), Slice(6,9), Slice(), Slice()}).argmax(1);
		std::cout<<"CerberusPoint 8.05"<<std::endl;

		CaloHitList caloHitListChangeU;
		CaloHitList caloHitListChangeV;
		CaloHitList caloHitListChangeW;

		PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, Backtracing(outputU, caloHitListChangeU, minX, minZ_U, TPC_VIEW_U, vertU));
		PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, Backtracing(outputV, caloHitListChangeV, minX, minZ_V, TPC_VIEW_V, vertV));
		PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, Backtracing(outputW, caloHitListChangeW, minX, minZ_W, TPC_VIEW_W, vertW));


		PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PopulateAvailabilityTensor(availability, caloHitVectorU, 0, minX, minZ_U));
		PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PopulateAvailabilityTensor(availability, caloHitVectorV, 1, minX, minZ_V));
		PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PopulateAvailabilityTensor(availability, caloHitVectorW, 2, minX, minZ_W));
		torch::save(availability, torch::str("/uboone/app/users/jdetje/PanLee_v08_57_00/DeepTesting/CerberusAvailability.pt"));


		PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, CaloHitReallocation(outputU, caloHitListChangeU, minX, minZ_U, TPC_VIEW_U, vertU));
		PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, CaloHitReallocation(outputV, caloHitListChangeV, minX, minZ_V, TPC_VIEW_V, vertV));
		PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, CaloHitReallocation(outputW, caloHitListChangeW, minX, minZ_W, TPC_VIEW_W, vertW));


		torch::Tensor pandoraRecoPost = torch::zeros({1,3,IMSIZE,IMSIZE}, torch::kFloat32); //Creates the data tensor
		PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PopulatePandoraReconstructionTensor(pandoraRecoPost, pPfoList, TPC_VIEW_U, 0, minX, minZ_U));
		PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PopulatePandoraReconstructionTensor(pandoraRecoPost, pPfoList, TPC_VIEW_V, 1, minX, minZ_V));
		PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PopulatePandoraReconstructionTensor(pandoraRecoPost, pPfoList, TPC_VIEW_W, 2, minX, minZ_W));
		torch::save(pandoraRecoPost, torch::str("/uboone/app/users/jdetje/PanLee_v08_57_00/DeepTesting/CerberusPandoraRecoPost.pt")); //Test_jdetje/



		std::cout<<"CerberusPoint 9"<<std::endl;
		return STATUS_CODE_SUCCESS;
	}

	StatusCode CerberusAlgorithm::Backtracing(const torch::Tensor &tensor, CaloHitList &caloHitListChange, const float minX, const float minZ, const HitType tpcView, const CartesianVector ShowerVertex2D)
	{
	    const ClusterList *pClusterList(nullptr);
	    PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::GetCurrentList(*this, pClusterList));
    	const PfoList *pPfoList(nullptr);
		PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::GetCurrentList(*this, pPfoList));

  //      	const ClusterList originalClusterList(*pClusterList);
  //       std::string originalClustersListName;
  //       //PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::InitializeReclustering(*this, Tracks(), originalClusterList, originalClustersListName));

	 //    const ClusterList* pNewClusterList(pClusterList);
  //       std::string newClusterListName(originalClustersListName);
  //       PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::CreateTemporaryListAndSetCurrent(*this, pNewClusterList, newClusterListName));
		// //PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, this->GetManager<CaloHit>()->PrepareForClustering(*this, newClusterListName));

        for (const ParticleFlowObject *const pPfo : *pPfoList)
		{	
			std::cout<<"++++ ++++ New Pfo ++++  ++++ "<<std::endl;
			ClusterList clusterList;
			LArPfoHelper::GetClusters(pPfo, tpcView, clusterList);
			const bool isShower = LArPfoHelper::IsShower(pPfo);
			const bool neutrinoFinalState = LArPfoHelper::IsNeutrinoFinalState(pPfo);
		   	for (const Cluster *const pCluster : clusterList)
	    	{
				std::cout<<"---- ---- New Cluster ----  ---- "<<std::endl;
				CaloHitList caloHitList;
	    		pCluster->GetOrderedCaloHitList().FillCaloHitList(caloHitList);
		    	for (const CaloHit *const pCaloHit : caloHitList)
				{
					int x, z;
					if(!inViewXZ(x, z, pCaloHit, minX, minZ)) continue; // Sets x, z for hits that are in the crop area 
					std::cout<<ShowerVertex2D.GetX();// TODO: Remove this !!!!!!
					const int caloHitClass = tensor.index({0, x, z}).item<int>();
					std::cout<<"|"<<caloHitClass;
					if((!isShower && caloHitClass==0)||(isShower && caloHitClass==1)||(neutrinoFinalState && caloHitClass==2)||(!neutrinoFinalState && caloHitClass!=2))
					{
						std::cout<<"### CerberusAlgorithm::Backtracing: Point 4.02"<<std::endl;
    					caloHitListChange.push_back(pCaloHit);

    					CaloHitList caloHitListUpdated;
			    		pCluster->GetOrderedCaloHitList().FillCaloHitList(caloHitListUpdated);
						//std::cout<<"### CerberusAlgorithm::Backtracing: Point 4.1 - CaloHitChanged - caloHit available: "<<PandoraContentApi::IsAvailable(*this, pCaloHit)<<" caloHitList.size(): "<<caloHitListUpdated.size()<<" bestCaloHitList.size()"<<bestCaloHitList.size()<<std::endl;
						
						if(caloHitListUpdated.size()==1)
						{
							//PANDORA_THROW_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::MergeAndDeleteClusters(*this, pBestCluster, pCluster)); // pBestCluster is enlarged and pCluster is deleted
							//PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::Delete(*this, pPfo)); // TODO this can be activated twice !!!!!!!!!!!!!!!!!!!!!!!!!!
							PANDORA_THROW_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::RemoveFromPfo(*this, pPfo, pCluster));
							//PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::Delete(*this, pCluster));
							break;
						}

						PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::RemoveFromCluster(*this, pCluster, pCaloHit));	
					}
				}
			}
		}


     //    if (!pNewClusterList->empty())
    	// {
	    // 	PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::SaveList<Cluster>(*this, m_caloHitListNames[0]));
     //    	PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::ReplaceCurrentList<Cluster>(*this, m_caloHitListNames[0]));
     //    }
	    
	    return STATUS_CODE_SUCCESS;
	}

	bool CerberusAlgorithm::inViewXZ(int &x, int &z, const CaloHit *const pCaloHit, const float minX, const float minZ)
	{
		const CartesianVector vec = pCaloHit->GetPositionVector();
		x = (int)((vec.GetX()-minX)/0.3f);
		z = (int)((vec.GetZ()-minZ)/0.3f);
		if(x>=IMSIZE || z>=IMSIZE || x<0 || z<0) return false; // Hits that are not in the crop area
		return true;
	}

	StatusCode CerberusAlgorithm::CaloHitReallocation(const torch::Tensor &tensor, const CaloHitList &caloHitListChange, const float minX, const float minZ, const HitType tpcView, const CartesianVector ShowerVertex2D)
	{
		// TODO: Replace with updated cluster list !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	    const ClusterList *pClusterList(nullptr);
	    PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::GetCurrentList(*this, pClusterList));
	    std::cout<<"PPP CerberusAlgorithm::CaloHitReallocation - pClusterList.size(): "<<pClusterList->size()<<std::endl;

		for (const CaloHit *const pCaloHit : caloHitListChange)
		{
			int x, z;
			if(!inViewXZ(x, z, pCaloHit, minX, minZ)) return STATUS_CODE_FAILURE; //Also sets x,z value 
			const int caloHitClass = tensor.index({0, x, z}).item<int>();
			const Cluster *pBestCluster(nullptr);
			FindSuitableCluster(pCaloHit, pBestCluster, caloHitClass, 150);
			if (pBestCluster) 
			{
				const bool availability = PandoraContentApi::IsAvailable(*this, pCaloHit);
				const HitType ht  = LArClusterHelper::GetClusterHitType(pBestCluster);
				std::cout<<"QQQ CerberusAlgorithm::CaloHitReallocation - availability: "<<availability<<" - HitType ht: "<<ht<<" - tpcView: "<<tpcView<<std::endl;
				if(!availability) continue;
				PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::AddToCluster(*this, pBestCluster, pCaloHit)); // Meaning: if it is not a null pointer add the hit to the cluster
				//caloHitListChange.remove(pCaloHit);
			}
		}
	    return STATUS_CODE_SUCCESS;
	}

	// StatusCode CerberusAlgorithm::CaloHitReallocation(const torch::Tensor &tensor, const CaloHitList &caloHitListChange, const float minX, const float minZ, const pandora::HitType tpcView, const CartesianVector ShowerVertex2D)
	// {
	// 	// TODO: Replace with updated cluster list !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	//     const ClusterList *pClusterList(nullptr);
	//     PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::GetCurrentList(*this, pClusterList));
	    
	//     const CaloHitList *pCaloHitList(nullptr);
	//     PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::GetCurrentList(*this, pCaloHitList));

	// 	for (const CaloHit *const pCaloHit : pCaloHitList)
	// 	{
	// 		const bool availability = PandoraContentApi::IsAvailable(*this, pCaloHit);
	// 		if(!availability) continue;

	// 		const CartesianVector vec = pCaloHit->GetPositionVector();
	// 		const int x = (int)((vec.GetX()-minX)/0.3f);
	// 		const int z = (int)((vec.GetZ()-minZ)/0.3f);
	// 		if(x>=IMSIZE || z>=IMSIZE || x<0 || z<0) return STATUS_CODE_FAILURE;
	// 		const int caloHitClass = tensor.index({0, x, z}).item<int>();

	// 		const Cluster *pBestCluster(nullptr);
	// 		FindSuitableCluster(pCaloHit, pBestCluster, caloHitClass, 10);
	// 		if (pBestCluster)
	// 		{
	// 			PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::AddToCluster(*this, pBestCluster, pCaloHit)); // Meaning: if it is not a null pointer add the hit to the cluster
	// 		} 
	// 		else if(caloHitClass!=2)
	// 		{
	// 			const Cluster *pCluster(nullptr);
	// 			PandoraContentApi::Cluster::Parameters parameters;
	// 			parameters.m_caloHitList.push_back(pCaloHit);
	// 			PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::Cluster::Create(*this, parameters, pCluster));
	// 			PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::GetCurrentList(*this, pClusterList));
	// 		}

	// 	}
	//     return STATUS_CODE_SUCCESS;
	// }


	// StatusCode CerberusAlgorithm::ClusterCreation(const torch::Tensor &tensor, const CaloHitList &caloHitListChange, const float minX, const float minZ, const pandora::HitType tpcView, const CartesianVector ShowerVertex2D)
	// {
	// 	// TODO: Replace with updated cluster list !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	//     const ClusterList *pClusterList(nullptr);
	//     PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::GetCurrentList(*this, pClusterList));

	// 	for (const CaloHit *const pCaloHit : caloHitListChange)
	// 	{
	// 		const CartesianVector vec = pCaloHit->GetPositionVector();
	// 		const int x = (int)((vec.GetX()-minX)/0.3f);
	// 		const int z = (int)((vec.GetZ()-minZ)/0.3f);
	// 		if(x>=IMSIZE || z>=IMSIZE || x<0 || z<0) return STATUS_CODE_FAILURE;
	// 		const int caloHitClass = tensor.index({0, x, z}).item<int>();

	// 		const bool availability = PandoraContentApi::IsAvailable(*this, pCaloHit);
	// 		const Cluster *pBestCluster(nullptr);
	// 		FindSuitableCluster(pCaloHit, pBestCluster, caloHitClass, 10);
	// 		if (pBestCluster) PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::AddToCluster(*this, pBestCluster, pCaloHit)); // Meaning: if it is not a null pointer add the hit to the cluster
	// 	}

	// 	if()
	// 	{
	// 		const Cluster *pCluster(nullptr);
	// 		PandoraContentApi::Cluster::Parameters parameters;
	// 		parameters.m_caloHitList.push_back(pCaloHit);
	// 		PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::Cluster::Create(*this, parameters, pCluster));
	// 	}
		
	//     return STATUS_CODE_SUCCESS;
	// }	


	// StatusCode CerberusAlgorithm::PopulateAvailabilityTensor(torch::Tensor &tensor, const CaloHitVector &caloHitVector, const int index, const float minX, const float minZ) // index 0: U-View, 1: V-View, 2: W-View
	// {
	// 	for (const CaloHit *const pCaloHit : caloHitVector)
	// 	{
	// 		const int x = (int)((pCaloHit->GetPositionVector().GetX()-minX)/0.3f);
	// 		const int z = (int)((pCaloHit->GetPositionVector().GetZ()-minZ)/0.3f);
	// 		if(x>=IMSIZE || z>=IMSIZE || x<0 || z<0) continue; // Skipps hits that are not in the crop area



	// 		const float value = tensor.index({0, index, x, z}).item<float>();
	// 		tensor.index_put_({0, index, x, z}, value+availability);
	// 	}
	// 	return STATUS_CODE_SUCCESS;
	// }


	// https://github.com/PandoraPFA/LArContent/blob/d4e5aa8b34cae1809f24c1f61d1d2ed0d7994096/larpandoracontent/LArHelpers/LArPfoHelper.cc
	StatusCode CerberusAlgorithm::FindSuitableCluster(const CaloHit *const pCaloHit, const Cluster *&pBestCluster, const int caloHitClass, const float maxDistance)
	{
		const HitType tpcView(pCaloHit->GetHitType());
		const ClusterList *pInputClusterList(nullptr);
		ClusterList clusterList;
		PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::GetCurrentList(*this, pInputClusterList));
	    LArClusterHelper::GetClustersByHitType(*pInputClusterList, tpcView, clusterList);
	    
	    float closestDistanceSquared(maxDistance * maxDistance);
	    const CartesianVector positionVector(pCaloHit->GetPositionVector());
	    
	    std::cout<<"!!! CerberusAlgorithm::FindClosestTrackCluster: Point 2"<<std::endl;
	    std::cout<<"PPP CerberusAlgorithm::FindClosestTrackCluster - pClusterList.size(): "<<clusterList.size()<<std::endl;
	    for (const Cluster *const pCandidateCluster : clusterList)
	    {
	    	const int pdgCode = pCandidateCluster->GetParticleId();
	    	if((pdgCode==E_MINUS && caloHitClass!=0) || (pdgCode==MU_MINUS && caloHitClass!=1)) continue; //TODO: Add Cosmic neutrino discrimination 
	    	std::cout<<"!!! CerberusAlgorithm::FindClosestTrackCluster: Point 5"<<std::endl;
	        const CartesianVector candidateCentroid(pCandidateCluster->GetCentroid(pCandidateCluster->GetInnerPseudoLayer()));
	        //const float distanceSquared((positionVector - candidateCentroid).GetMagnitudeSquared());
	        const float distanceSquared = positionVector.GetDistanceSquared(candidateCentroid);
	    	
	    	const CartesianVector clusterDirection(pCandidateCluster->GetInitialDirection()); // GetDirection?????
	    	const CartesianVector hitDirection(positionVector-candidateCentroid);
	        const float theta = clusterDirection.GetOpeningAngle(hitDirection);

    		std::ofstream file("/uboone/app/users/jdetje/PanLee_v08_57_00/DeepTesting/pos.bin", std::ios::out | std::ios::binary | std::ios::app); 
			std::array<int, 8> pos = {0};
			pos[0] = (int) candidateCentroid.GetX();
			pos[1] = (int) candidateCentroid.GetZ();
			pos[2] = (int) clusterDirection.GetX();
			pos[3] = (int) clusterDirection.GetZ();

			pos[4] = (int) positionVector.GetX();
			pos[5] = (int) positionVector.GetZ();
			pos[6] = (int) hitDirection.GetX();
			pos[7] = (int) hitDirection.GetZ();
			file.write((char*)&pos, sizeof(pos));
			file.close();


	        std::cout<<"!!! CerberusAlgorithm::FindClosestTrackCluster: Point 6 - distanceSquared" <<distanceSquared<<" - theta: "<<theta<<std::endl;
	        if (distanceSquared < closestDistanceSquared && ((theta<M_PI/15.f || theta>M_PI*(1-1/15.f)) || distanceSquared<50.f*50.f))
	        {
	            closestDistanceSquared = distanceSquared;
	            pBestCluster = pCandidateCluster;
	    		std::cout<<"!!! CerberusAlgorithm::FindClosestTrackCluster: Point 7"<<std::endl;
	        }
	    }
	    return STATUS_CODE_SUCCESS;
	}

	StatusCode CerberusAlgorithm::PopulatePandoraReconstructionTensor(torch::Tensor &tensor, const PfoList *const pPfoList, const HitType tpcView, const int index, const float minX, const float minZ) // index 0: U-View, 1: V-View, 2: W-View
	{

        for (const ParticleFlowObject *const pPfo : *pPfoList)
		{	
			ClusterList clusterList;
			LArPfoHelper::GetClusters(pPfo, tpcView, clusterList);
			int value(3);
			
			if(!LArPfoHelper::IsNeutrinoFinalState(pPfo)) value=3;
			else
			{
				if(LArPfoHelper::IsShower(pPfo)) value = 1;
				else value = 2;
			}
		   	for (const Cluster *const pCluster : clusterList)
	    	{
				CaloHitList caloHitList;
	    		pCluster->GetOrderedCaloHitList().FillCaloHitList(caloHitList);
		    	for (const CaloHit *const pCaloHit : caloHitList)
				{
					const int x = (int)((pCaloHit->GetPositionVector().GetX()-minX)/0.3f);
					const int z = (int)((pCaloHit->GetPositionVector().GetZ()-minZ)/0.3f);
					if(x>=IMSIZE || z>=IMSIZE || x<0 || z<0) continue; // Skipps hits that are not in the crop area
					tensor.index_put_({0, index, x, z}, value);
				}
			}
		}
		return STATUS_CODE_SUCCESS;
	}

	StatusCode CerberusAlgorithm::PopulateAvailabilityTensor(torch::Tensor &tensor, const CaloHitVector &caloHitVector, const int index, const float minX, const float minZ) // index 0: U-View, 1: V-View, 2: W-View
	{
		for (const CaloHit *const pCaloHit : caloHitVector)
		{
			const int x = (int)((pCaloHit->GetPositionVector().GetX()-minX)/0.3f);
			const int z = (int)((pCaloHit->GetPositionVector().GetZ()-minZ)/0.3f);

			if(x>=IMSIZE || z>=IMSIZE || x<0 || z<0) continue; // Skipps hits that are not in the crop area

			const int availability = (int) PandoraContentApi::IsAvailable(*this, pCaloHit);
			const float value = tensor.index({0, index, x, z}).item<float>();
			tensor.index_put_({0, index, x, z}, value+availability);
		}
		return STATUS_CODE_SUCCESS;
	}

	StatusCode CerberusAlgorithm::PopulateMCTensor(torch::Tensor &tensor, const CaloHitVector &caloHitVector, const int index, const float minX, const float minZ)
	{
		float value(0.f);
		for (const CaloHit *const pCaloHit : caloHitVector)
		{
			int x, z;
			if(!inViewXZ(x, z, pCaloHit, minX, minZ)) continue; // Skipps hits that are not in the crop area
			std::array<float, 2> pixel = {0};
			const MCParticleWeightMap  &mcParticleWeightMap(pCaloHit->GetMCParticleWeightMap());
			// Populates prediction image
			//std::cout<<"--------------------- New Hit"<<std::endl;
			for (const MCParticleWeightMap::value_type &mapEntry : mcParticleWeightMap)
			{
				const int particleID = mapEntry.first->GetParticleId();
				//std::cout<<"--------------------- particleID: "<<particleID<<" mapEntry.second"<<mapEntry.second<<std::endl;
				switch(particleID)
				{
					case 22: case 11: case -11:
						pixel[0] += mapEntry.second;
						break;
					case 2212:
						pixel[1] += mapEntry.second;
						break;
				}
			}

			if(pixel[0]+pixel[1]<0.1) value=3.f;
			else
			{
				if(pixel[0]>pixel[1]) value=1.f;
				else value = 2.f;
			}
			tensor.index_put_({0, index, x, z}, value);
		}

		return STATUS_CODE_SUCCESS;
	}	


	void CerberusAlgorithm::FillMinimizationArray(std::array<float, SEG> &hitDensity, const PfoList *const pPfoList, const CaloHitList *const pCaloHitList, const CartesianVector v, const float startD1, const float startD2, const bool directionX, const HitType tpcView)
	{
		float weight, d1, d2;

		for (const ParticleFlowObject *const pPfo : *pPfoList) // Finds and adds shower to pfoListCrop
		{
			if (LArPfoHelper::IsShower(pPfo)) // && LArPfoHelper::IsNeutrinoFinalState(pPfo)
			{
				if(LArPfoHelper::IsNeutrinoFinalState(pPfo)) weight = 1.f;
				else weight = 0.f;
			}
			else
			{
				if(LArPfoHelper::IsNeutrinoFinalState(pPfo)) weight = 2.f;
				else weight = 0.f;
			}

			try
			{
				CartesianVector v2 =  LArPfoHelper::GetVertex(pPfo)->GetPosition();
				v2 = LArGeometryHelper::ProjectPosition(this->GetPandora(), v2, tpcView); // Project 3D vertex onto 2D view
				const float xDiff = v.GetX()-v2.GetX();
				const float zDiff = v.GetZ()-v2.GetZ();
				const float squaredDist = xDiff*xDiff+zDiff*zDiff;
				if(squaredDist>2000) weight *= 1.f;//6000.0/(squaredDist+4000.0);
				//std::cout<<"Weight: "<<weight<<"  sqdst: "<<squaredDist<<std::endl;
			} 
				catch(StatusCodeException &statusCodeException)
			{
				std::cout<<"CerberusAlgorithm::FillMinimizationArray: No Pfo Vertex Found"<<std::endl;
			}


			PfoList pfoListTemp;
			pfoListTemp.push_back(pPfo);
			CaloHitList caloHitList;
			LArPfoHelper::GetCaloHits(pfoListTemp, tpcView, caloHitList);
			for (const CaloHit *const pCaloHit : caloHitList)
			{
				if(directionX){
					d1 = pCaloHit->GetPositionVector().GetX();
					d2 = pCaloHit->GetPositionVector().GetZ();
				} else {
					d1 = pCaloHit->GetPositionVector().GetZ();
					d2 = pCaloHit->GetPositionVector().GetX();
				}
				const int pixel = static_cast<int>(((d1-startD1)/0.3f + IMSIZE)/(2.0*IMSIZE)*SEG);
				if(pixel>=0 && pixel<SEG && (d2-startD2)/0.3<IMSIZE && (d2-startD2)>=0)
					hitDensity[pixel]+=weight;
			}
		}
		weight = 0.2f;
		for (const CaloHit *const pCaloHit : *pCaloHitList)
		{
			if(!PandoraContentApi::IsAvailable(*this, pCaloHit))
			{	
				if(directionX){
					d1 = pCaloHit->GetPositionVector().GetX();
					d2 = pCaloHit->GetPositionVector().GetZ();
				} else {
					d1 = pCaloHit->GetPositionVector().GetZ();
					d2 = pCaloHit->GetPositionVector().GetX();
				}
				const int pixel = static_cast<int>(((d1-startD1)/0.3f + IMSIZE)/(2.0*IMSIZE)*SEG);
				if(pixel>=0 && pixel<SEG && (d2-startD2)/0.3<IMSIZE && (d2-startD2)>=0)
					hitDensity[pixel]+=weight;
			}
		}
	}

	float CerberusAlgorithm::FindMin(const std::array<float, SEG> hitDensity, const float startPoint) const
	{
		float total(0.f);
		int best = 0;
		for(int i=0; i<SEG/2; i++)
			{
				const int j = SEG/2+i;
				total += hitDensity[j]-hitDensity[i];
				if(total>0.f)
				{
					best = i;
					total = 0.f;
				}
			}

		return ((2.0*best)/SEG-1) * IMSIZE * 0.3f + startPoint;
	}


	StatusCode CerberusAlgorithm::WriteDetectorGaps(torch::Tensor &tensor, const float minZ_U, const float minZ_V, const float minZ_W)
	{
		float minZ(0.f);
		for (const DetectorGap *const pDetectorGap : this->GetPandora().GetGeometry()->GetDetectorGapList())
		{
			const LineGap *const pLineGap = dynamic_cast<const LineGap*>(pDetectorGap);
        	if (!pLineGap) throw StatusCodeException(STATUS_CODE_INVALID_PARAMETER);

			const int gapType = static_cast<int>(pLineGap->GetLineGapType());
			
			switch(gapType)
			{
			case TPC_WIRE_GAP_VIEW_U: //gapType==0
				minZ = minZ_U;
				break;
			case TPC_WIRE_GAP_VIEW_V: //gapType==1
				minZ = minZ_V;
				break;
			case TPC_WIRE_GAP_VIEW_W: //gapType==2
				minZ = minZ_W;
				break;
			default:
				std::cout<<"Undeclared linegap type in CerberusAlgorithm::WriteDetectorGaps." <<std::endl;
				return STATUS_CODE_FAILURE;
			}

			const int gapStart = std::max(0,(int)((pLineGap->GetLineStartZ()-minZ)/0.3f));
			const int gapEnd = std::min(IMSIZE-1,(int)((pLineGap->GetLineEndZ()-minZ)/0.3f));
			tensor.index_put_({0, 2*gapType, Slice(gapStart,gapEnd), Slice()},1.f);
		}
		return STATUS_CODE_SUCCESS;
	}


	StatusCode CerberusAlgorithm::PopulateImage(torch::Tensor &tensor, const CaloHitVector &caloHitVector, const int index, const float minX, const float minZ) // index 0: U-View, 1: V-View, 2: W-View
	{
		for (const CaloHit *const pCaloHit : caloHitVector)
		{
			const int x = (int)((pCaloHit->GetPositionVector().GetX()-minX)/0.3f);
			const int z = (int)((pCaloHit->GetPositionVector().GetZ()-minZ)/0.3f);

			if(x>=IMSIZE || z>=IMSIZE || x<0 || z<0) continue; // Skipps hits that are not in the crop area

			float energy = pCaloHit->GetHadronicEnergy()/0.015; // Same normalisation that was used for training the TensorFlow model in python
			if(energy>1.f) energy=1.f;
			tensor.index_put_({0, 1+2*index, x, z}, energy);
		}
		return STATUS_CODE_SUCCESS;
	}	


//------------------------------------------------------------------------------------------------------------------------------------------
	StatusCode CerberusAlgorithm::ReadSettings(const TiXmlHandle xmlHandle)
	{
		// Read settings from xml file here
		// PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, XmlHelper::ReadValue(xmlHandle, "PfoListNames", m_pfoListNames));

		PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadVectorOfValues(xmlHandle,
	        "PfoListNames", m_pfoListNames));
		PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadVectorOfValues(xmlHandle,
	        "CaloHitListNames", m_caloHitListNames));
		// PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadVectorOfValues(xmlHandle,
	 //        "ClusterListNames", m_clusterListNames));

	    if (m_caloHitListNames.empty())
	    {
	        std::cout << "CerberusAlgorithm::ReadSettings - Must provide names of caloHit lists for use in U-Net." << std::endl;
	        return STATUS_CODE_INVALID_PARAMETER;
	    }

	   	// if (m_clusterListNames.empty())
	    // {
	    //     std::cout << "CerberusAlgorithm::ReadSettings - Must provide names of cluster lists for use in U-Net." << std::endl;
	    //     return STATUS_CODE_INVALID_PARAMETER;
	    // }

		return STATUS_CODE_SUCCESS;
	}

} // namespace lar_content

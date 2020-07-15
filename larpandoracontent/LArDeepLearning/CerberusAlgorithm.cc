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
				if(totalHits>20)
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

		///////////////////////////////////////////////////////////////////////////////////////
		/// Find minZ in U-view
		hitDensity= {0}; // Always combining 8 wires
		FillMinimizationArray(hitDensity, pPfoList, pCaloHitListU, vertU, vertU.GetZ(), minX, false, TPC_VIEW_U);
		minZ_U = FindMin(hitDensity, vertU.GetZ());

		///////////////////////////////////////////////////////////////////////////////////////
		/// Find minZ in V-view
		hitDensity= {0}; // Always combining 8 wires
		FillMinimizationArray(hitDensity, pPfoList, pCaloHitListV, vertV, vertV.GetZ(), minX, false, TPC_VIEW_V);
		minZ_V = FindMin(hitDensity, vertV.GetZ());

		///////////////////////////////////////////////////////////////////////////////////////
		/// Find minZ in W-view
		hitDensity= {0}; // Always combining 8 wires
		FillMinimizationArray(hitDensity, pPfoList, pCaloHitListW, vertW, vertW.GetZ(), minX, false, TPC_VIEW_W);
		minZ_W = FindMin(hitDensity, vertW.GetZ());

		///////////////////////////////////////////////////////////////////////////////////////
		/// Populate input tensor to Cerberus network
		torch::Tensor tensor = torch::zeros({1,6,IMSIZE,IMSIZE}, torch::kFloat32); //Creates the data tensor
		PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, WriteDetectorGaps(tensor, minZ_U, minZ_V, minZ_W));
		PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PopulateImage(tensor, caloHitVectorU, 0, minX, minZ_U));
		PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PopulateImage(tensor, caloHitVectorV, 1, minX, minZ_V));
		PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PopulateImage(tensor, caloHitVectorW, 2, minX, minZ_W));

		std::cout<<"CerberusPoint 7"<<std::endl;
		// ###### Load Torch model ######
		torch::jit::script::Module module;
		std::cout<<"CerberusPoint 7.1"<<std::endl;
		try {
			std::cout<<"CerberusPoint 7.2"<<std::endl;
			// Deserialize the ScriptModule from a file using torch::jit::load().
			module = torch::jit::load("/home/philip/Documents/Pandora5/PandoraPFA/LArContent-v03_16_02/larpandoracontent/LArDeepLearning/traced_resnet_model_CerberusU2_Jul10.pt");
			std::cout<<"CerberusPoint 7.3"<<std::endl;
		}
		catch (const c10::Error& e) {
			std::cout << "CerberusAlgorithm::Run() - Could not load Torch model"<<std::endl;
			return STATUS_CODE_FAILURE;
		}

		std::cout<<"CerberusPoint 8"<<std::endl;
		std::vector<torch::jit::IValue> inputs;
		inputs.push_back(tensor);
		at::Tensor output = module.forward(inputs).toTensor();
		output = output.argmax(1);
		PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, Backtracing(output, minX, minZ_U));
		std::cout<<"CerberusPoint 9"<<std::endl;
		return STATUS_CODE_SUCCESS;
	}

	StatusCode CerberusAlgorithm::Backtracing(const torch::Tensor &tensor, const float minX, const float minZ)
	{
		 // Make a number of cluster fragmentation operations, each taking an input cluster from the current list and using any number
	    // of clustering algorithms, defined in the PandoraSettings xml file, to reconfigure the constituent calo hits.
	    const ClusterList *pClusterList(nullptr);
	    PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::GetCurrentList(*this, pClusterList));

        	    
		// The reclustering mechanism allows the original and many attempts at new candidate clusters to exist concurrently, with the
        // calo hit availability monitoring functioning separately for each of the (potentially many) different hit configuration options.
       	const ClusterList originalClusterList(*pClusterList);
        std::string originalClustersListName;
        //PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::InitializeReclustering(*this, Tracks(), originalClusterList, originalClustersListName));

	    // Here, iterate over a local copy of the cluster list
	    const ClusterList* pNewClusterList(pClusterList);
        std::string newClusterListName(originalClustersListName);
        PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::CreateTemporaryListAndSetCurrent(*this, pNewClusterList, newClusterListName));
		//PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, this->GetManager<CaloHit>()->PrepareForClustering(*this, newClusterListName));


	    for (const Cluster *const pCluster : *pNewClusterList)
	    {
			CaloHitList caloHitList;
    		pCluster->GetOrderedCaloHitList().FillCaloHitList(caloHitList);
	    	for (const CaloHit *const pCaloHit : caloHitList)
			{
				const CartesianVector vec = pCaloHit->GetPositionVector();
				const int x = (int)((vec.GetX()-minX)/0.3f);
				const int z = (int)((vec.GetZ()-minZ)/0.3f);
				if(x>=IMSIZE || z>=IMSIZE || x<0 || z<0) continue; // Skipps hits that are not in the crop area

				const int caloHitClass = tensor.index({0, x, z}).item<int>();
				std::cout<<"CerberusAlgorithm: Point 1: "<<pCluster->GetParticleId()<<std::endl;
				if(pCluster->GetParticleId()==E_MINUS) // Is it a shower?
				{
					if(caloHitClass==0)
					{
						float shortestDistance = std::numeric_limits<float>::max();
						const Cluster * pbestCluster(nullptr);
						for (const Cluster *const pCluster2 : *pNewClusterList)
	    				{
	    					if(pCluster2->GetParticleId()==E_MINUS) continue; // Skip showers
							const float distance = LArClusterHelper::GetClosestDistance(vec, pCluster2);
							if(distance<shortestDistance)
							{
								shortestDistance=distance;
								pbestCluster = pCluster2;
							}
						}
						if(shortestDistance<10.f) //TODO: replace placeholder value 10.f
						{
							PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::RemoveFromCluster(*this, pCluster, pCaloHit));
							PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::AddToCluster(*this, pbestCluster, pCaloHit));
						}
					}
				}
				else 
				{
					if(caloHitClass==1 || caloHitClass==2)
					{
						std::cout<<"CerberusAlgorithm::Backtracing else if(caloHitClass==1 || caloHitClass==2)"<<std::endl;
					}
				}
			}
	    }

	    //PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::GetCurrentList(pNewClusterList, newClusterListName));
        //PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::EndReclustering(*this, newClusterListName));
        if (!pNewClusterList->empty())
    	{
	    	PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::SaveList<Cluster>(*this, m_caloHitListNames[0]));
        	PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::ReplaceCurrentList<Cluster>(*this, m_caloHitListNames[0]));
        }
	    return STATUS_CODE_SUCCESS;
	}

	void CerberusAlgorithm::FillMinimizationArray(std::array<float, SEG> &hitDensity, const PfoList *const pPfoList, const CaloHitList *const pCaloHitList, const CartesianVector v, const float startD1, const float startD2, const bool directionX, const HitType TPC_VIEW)
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
				v2 = LArGeometryHelper::ProjectPosition(this->GetPandora(), v2, TPC_VIEW); // Project 3D vertex onto 2D view
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
			LArPfoHelper::GetCaloHits(pfoListTemp, TPC_VIEW, caloHitList);
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

			const float energy = pCaloHit->GetHadronicEnergy(); // Populates input image
			tensor.index_put_({0, 1+index, x, z}, energy);
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

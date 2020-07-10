/**
 *  @file   larpandoracontent/LArWorkshop/CerberusAlgorithm.h
 *
 *  @brief  Header file for the Cerberus algorithm class.
 *
 *  $Log: $ 
 */

#ifndef LAR_CERBERUS_ALGORITHM_H
#define LAR_CERBERUS_ALGORITHM_H 1
#include "Pandora/Algorithm.h"
#include <torch/script.h>

#define IMSIZE 384 //256*1.5 Size of the generated image arrays
#define SEG 128

namespace lar_content
{
/** 
 *  @brief  CerberusAlgorithm class 
 */

	class CerberusAlgorithm : public pandora::Algorithm
	{
	public:
/**     
 *  @brief  Factory class for instantiating algorithm     
 */

		class Factory : public pandora::AlgorithmFactory
		{
		public:
			pandora::Algorithm *CreateAlgorithm() const;
		};

	private:
		pandora::StatusCode Run();
		pandora::StatusCode ReadSettings(const pandora::TiXmlHandle xmlHandle);
		pandora::StatusCode WriteDetectorGaps(torch::Tensor &tensor, const float minZ_U, const float minZ_V, const float minZ_W);
		pandora::StatusCode PopulateImage(torch::Tensor &tensor, const pandora::CaloHitVector &caloHitVector, const int index, const float minX, const float minZ);
		void fillMinimizationArray(std::array<float, 128> &hitDensity, const pandora::PfoList *const pPfoList, const pandora::CaloHitList *const pCaloHitList, const pandora::CartesianVector v, const float startD1, const float startD2, const bool directionX, const pandora::HitType TPC_VIEW);
		float findMin(const std::array<float, 128> hitDensity, const float startPoint) const;
		// Member variables here
		pandora::StringVector m_pfoListNames;
		pandora::StringVector m_clusterListNames; 
	};
//------------------------------------------------------------------------------------------------------------------------------------------
	inline pandora::Algorithm *CerberusAlgorithm::Factory::CreateAlgorithm() const
	{
		return new CerberusAlgorithm();
	}

} // namespace lar_content
#endif // #ifndef LAR_CERBERUS_ALGORITHM_H
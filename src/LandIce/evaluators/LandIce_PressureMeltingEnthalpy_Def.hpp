/*
 * LandIce_PressureMeltingEnthalpy_Def.hpp
 *
 *  Created on: Jun 6, 2016
 *      Author: abarone
 */

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Phalanx_Print.hpp"

#include "LandIce_PressureMeltingEnthalpy.hpp"

namespace LandIce
{

template<typename EvalT, typename Traits, typename PressST, typename SurfTempST>
PressureMeltingEnthalpy<EvalT,Traits,PressST,SurfTempST>::
PressureMeltingEnthalpy(const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl):
  pressure       (p.get<std::string> ("Hydrostatic Pressure Variable Name"), dl->node_scalar),
  surfaceTemp    (p.get<std::string> ("Surface Air Temperature Name"), dl->node_scalar),
  meltingTemp    (p.get<std::string> ("Melting Temperature Variable Name"), dl->node_scalar),
  enthalpyHs     (p.get<std::string> ("Enthalpy Hs Variable Name"), dl->node_scalar),
  surfaceEnthalpy(p.get<std::string> ("Surface Air Enthalpy Name"), dl->node_scalar)
{
  std::vector<PHX::Device::size_type> dims;
  dl->node_qp_vector->dimensions(dims);

  numNodes = dims[1];

  this->addDependentField(pressure);
  this->addDependentField(surfaceTemp);

  this->addEvaluatedField(meltingTemp);
  this->addEvaluatedField(enthalpyHs);
  this->addEvaluatedField(surfaceEnthalpy);
  this->setName("Pressure-melting Enthalpy");

  // Setting parameters
  Teuchos::ParameterList& physics = *p.get<Teuchos::ParameterList*>("LandIce Physical Parameters");
  rho_i = physics.get<double>("Ice Density"); //916
  c_i   = physics.get<double>("Heat capacity of ice");  //2009
  T0    = physics.get<double>("Reference Temperature"); //265
  beta =  physics.get<double>("Clausius-Clapeyron Coefficient");
}

template<typename EvalT, typename Traits, typename PressST, typename SurfTempST>
void PressureMeltingEnthalpy<EvalT,Traits,PressST,SurfTempST>::
postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
{}

template<typename EvalT, typename Traits, typename PressST, typename SurfTempST>
void PressureMeltingEnthalpy<EvalT,Traits,PressST,SurfTempST>::
evaluateFields(typename Traits::EvalData d)
{
  const double powm6 = 1e-6; // [k^2], k=1000

  for (std::size_t cell = 0; cell < d.numCells; ++cell)
    for (std::size_t node = 0; node < numNodes; ++node) {
      meltingTemp(cell,node) = - beta * pressure(cell,node) + 273.15;
      enthalpyHs(cell,node) = rho_i * c_i * ( meltingTemp(cell,node) - T0 ) * powm6;
      surfaceEnthalpy(cell,node) = rho_i * c_i * ( std::min(surfaceTemp(cell,node),273.15) - T0 ) * powm6;
    }
}

} // namespace LandIce

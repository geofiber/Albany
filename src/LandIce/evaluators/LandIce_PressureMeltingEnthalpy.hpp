/*
 * LandIce_PressureMeltingEnthalpy.hpp
 *
 *  Created on: Jun 6, 2016
 *      Author: abarone
 */

#ifndef LANDICE_PRESSURE_MELTING_ENTHALPY_HPP
#define LANDICE_PRESSURE_MELTING_ENTHALPY_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"
#include "PHAL_Dimension.hpp"
#include "PHAL_Utilities.hpp"

namespace LandIce
{

/** \brief Pressure-melting enthalpy

  This evaluator computes enthalpy of the ice at pressure-melting temperature Tm(p).
 */

template<typename EvalT, typename Traits, typename PressST>
class PressureMeltingEnthalpy: public PHX::EvaluatorWithBaseImpl<Traits>,
                               public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  PressureMeltingEnthalpy (const Teuchos::ParameterList& p,
                           const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void evaluateFields(typename Traits::EvalData d);

private:
  typedef typename EvalT::ParamScalarT ParamScalarT;

  // Input:
  PHX::MDField<const PressST,Cell,Node>     pressure;    //[Pa], [kg m^{-1} s^{-2}]

  // Output:
  PHX::MDField<PressST,Cell,Node>      meltingTemp; //[K]
  PHX::MDField<PressST,Cell,Node>      enthalpyHs;       //[MW s m^{-3}]

  unsigned int numNodes;

  double c_i;   //[J Kg^{-1} K^{-1}], Heat capacity of ice
  double rho_i; //[kg m^{-3}]
  double T0;    //[K]
  double beta;  //[K Pa^{-1}]
  double Tm; //[K], 273.15
  double enthalpyHs_scaling;

  PHAL::MDFieldMemoizer<Traits> memoizer;

  public:

  typedef Kokkos::View<int***, PHX::Device>::execution_space ExecutionSpace;

  typedef Kokkos::RangePolicy< ExecutionSpace > PressureMeltingEnthalpy_Policy;

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& i) const;

};

} // Namespace LandIce


#endif // LANDICE_PRESSURE_MELTING_ENTHALPY_HPP

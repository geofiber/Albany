//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Phalanx_Print.hpp"

#include "LandIce_IceOverburden.hpp"

namespace LandIce {

template<typename EvalT, typename Traits>
IceOverburden<EvalT, Traits>::
IceOverburden (const Teuchos::ParameterList& p,
               const Teuchos::RCP<Albany::Layouts>& dl)
{
  // Check if it is a sideset evaluation
  eval_on_side = false;
  if (p.isParameter("Side Set Name")) {
    sideSetName = p.get<std::string>("Side Set Name");
    eval_on_side = true;
  }
  TEUCHOS_TEST_FOR_EXCEPTION (eval_on_side!=dl->isSideLayouts, std::logic_error,
      "Error! Input Layouts structure not compatible with requested field layout.\n");

  Teuchos::RCP<PHX::DataLayout> layout;
  if (p.isParameter("Nodal") && p.get<bool>("Nodal")) {
    layout = dl->node_scalar;
  } else {
    layout = dl->qp_scalar;
  }

  numPts = eval_on_side ? layout->extent(2) : layout->extent(1);

  H   = PHX::MDField<const RealType>(p.get<std::string> ("Ice Thickness Variable Name"), layout);
  P_o = PHX::MDField<RealType>(p.get<std::string> ("Ice Overburden Variable Name"), layout);

  this->addDependentField (H);
  this->addEvaluatedField (P_o);

  // Setting parameters
  Teuchos::ParameterList& physics  = *p.get<Teuchos::ParameterList*>("LandIce Physical Parameters");

  rho_i = physics.get<double>("Ice Density",1000);
  g     = physics.get<double>("Gravity Acceleration",9.8);

  this->setName("IceOverburden"+PHX::print<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void IceOverburden<EvalT, Traits>::
evaluateFields (typename Traits::EvalData workset)
{
  if (eval_on_side) {
    evaluateFieldsSide(workset);
  } else {
    evaluateFieldsCell(workset);
  }
}

template<typename EvalT, typename Traits>
void IceOverburden<EvalT, Traits>::
evaluateFieldsSide (typename Traits::EvalData workset)
{
  auto it_ss = workset.sideSets->find(sideSetName);

  if (it_ss==workset.sideSets->end()) {
    return;
  }

  const auto& sideSet = it_ss->second;
  for (const auto& it : sideSet) {
    // Get the local data of side and cell
    const int cell = it.elem_LID;
    const int side = it.side_local_id;

    for (unsigned int pt=0; pt<numPts; ++pt) {
      P_o (cell,side,pt) = rho_i*g*H(cell,side,pt);
    }
  }
}

//**********************************************************************
template<typename EvalT, typename Traits>
void IceOverburden<EvalT, Traits>::
evaluateFieldsCell (typename Traits::EvalData workset)
{
  for (unsigned int cell=0; cell<workset.numCells; ++cell) {
    for (unsigned int pt=0; pt<numPts; ++pt) {
      P_o (cell,pt) = rho_i*g*H(cell,pt);
    }
  }
}

} // Namespace LandIce

//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_FancyOStream.hpp"

#include "PHAL_FactoryTraits.hpp"
#include "Albany_Utils.hpp"
#include "Albany_BCUtils.hpp"
#include "LandIce_StokesFOHydrology.hpp"

namespace LandIce {

StokesFOHydrology::
StokesFOHydrology (const Teuchos::RCP<Teuchos::ParameterList>& params_,
                   const Teuchos::RCP<Teuchos::ParameterList>& discParams_,
                   const Teuchos::RCP<ParamLib>& paramLib_,
                   const int numDim_) :
  StokesFOBase(params_, discParams_, paramLib_, numDim_)
{
  // Figure out what kind of hydro problem we solve
  eliminate_h = params->sublist("LandIce Hydrology").get<bool>("Eliminate Water Thickness", false);
  has_h_till  = params->sublist("LandIce Hydrology").get<double>("Maximum Till Water Storage",0.0) > 0.0;
  has_p_dot   = params->sublist("LandIce Hydrology").get<double>("Englacial Porosity",0.0) > 0.0;

  std::string sol_method = params->get<std::string>("Solution Method");
  if (sol_method=="Transient") {
    unsteady = true;
  } else {
    unsteady = false;
  }

  TEUCHOS_TEST_FOR_EXCEPTION (eliminate_h && unsteady, std::logic_error,
                              "Error! Water Thickness can be eliminated only in the steady case.\n");
  TEUCHOS_TEST_FOR_EXCEPTION (has_h_till && !unsteady, std::logic_error,
                              "Error! Till Water Storage equation only makes sense in the unsteady case.\n");
  TEUCHOS_TEST_FOR_EXCEPTION (has_p_dot && !unsteady, std::logic_error,
                              "Error! Englacial porosity model only makes sense in the unsteady case.\n");

  // Fill the variables names
  auto& vnames = params->sublist("Variables Names");

  var_names.water_pressure             = vnames.get<std::string>("Water Pressure Name", "water_pressure");
  var_names.water_thickness            = vnames.get<std::string>("Water Thickness Name", "water_thickness");
  var_names.till_water_storage         = vnames.get<std::string>("Till Water Storage Name", "till_water_storage");
  var_names.water_pressure_dot         = vnames.get<std::string>("Water Pressure Dot Name", "water_pressure_dot");
  var_names.water_thickness_dot        = vnames.get<std::string>("Water Thickness Dot Name", "water_thickness_dot");
  var_names.till_water_storage_dot     = vnames.get<std::string>("Till Water Storage Dot Name", "till_water_storage_dot");

  var_names.hydraulic_potential        = vnames.get<std::string>("Hydraulic Potential Name", "hydraulic_potential");
  var_names.ice_softness               = vnames.get<std::string>("Ice Softness Name", flow_factor_name);
  var_names.ice_overburden             = vnames.get<std::string>("Ice Overburden Name", "ice_overburden");
  var_names.effective_pressure         = vnames.get<std::string>("Effective Pressure Name", effective_pressure_name);
  var_names.temperature                = vnames.get<std::string>("Temperature Name", temperature_name);
  var_names.corrected_temperature      = vnames.get<std::string>("Corrected Temperature Name", corrected_temperature_name);
  var_names.ice_thickness              = vnames.get<std::string>("Ice Thickness Name", ice_thickness_name);
  var_names.surface_height             = vnames.get<std::string>("Surface Height Name", surface_height_name);
  var_names.beta                       = vnames.get<std::string>("Beta Name", "beta");
  var_names.melting_rate               = vnames.get<std::string>("Melting Rate Name", "melting_rate");
  var_names.surface_water_input        = vnames.get<std::string>("Surface Water Input Name", "surface_water_input");
  var_names.surface_mass_balance       = vnames.get<std::string>("Surface Mass Balance Name", "surface_mass_balance");
  var_names.geothermal_flux            = vnames.get<std::string>("Geothermal Flux Name", "geothermal_flux");
  var_names.water_discharge            = vnames.get<std::string>("Water Discharge Name", "water_discharge");
  var_names.sliding_velocity           = vnames.get<std::string>("Sliding Velocity Name", "sliding_velocity");
  var_names.basal_grav_water_potential = vnames.get<std::string>("Basal Gravitational Water Potential Name", "basal_gravitational_water_potential");

  // Set the num PDEs depending on the problem specs
  if (eliminate_h) {
    hydro_neq = 1;
  } else if (has_h_till) {
    hydro_neq = 3;
  } else {
    hydro_neq = 2;
  }
  stokes_neq = vecDimFO;
  stokes_ndofs = 1;
  hydro_ndofs = hydro_neq;

  this->setNumEquations(hydro_neq + stokes_neq);
  rigidBodyModes->setParameters(neq, computeConstantModes, vecDimFO, computeRotationModes);

  hydro_dofs_names.resize(hydro_neq);
  hydro_resids_names.resize(hydro_neq);
  stokes_dofs_names.resize(stokes_neq);
  stokes_resids_names.resize(stokes_neq);

  stokes_dofs_names.deepCopy(dof_names());
  stokes_resids_names.deepCopy(resid_names());

  dof_names.resize(stokes_ndofs+hydro_neq);
  resid_names.resize(stokes_ndofs+hydro_neq);
  scatter_names.resize(2);

  // We always solve for the water pressure
  hydro_dofs_names[0]   = dof_names[stokes_ndofs] = var_names.water_pressure;
  hydro_resids_names[0] = resid_names[stokes_ndofs] = "Residual Mass Eqn";
  scatter_names[1] = "Scatter Hydrology";

  if (!eliminate_h) {
    hydro_dofs_names[1]   = dof_names[stokes_ndofs+1] = var_names.water_thickness;
    hydro_resids_names[1] = resid_names[stokes_ndofs+1] = "Residual Cavities Eqn";
  }

  if (has_h_till) {
    hydro_dofs_names[2]   = dof_names[stokes_ndofs+2] = var_names.till_water_storage;
    hydro_resids_names[2] = resid_names[stokes_ndofs+2] = "Residual Till Storage Eqn";
  }

  // Set the hydrology equations as side set equations on the basal side
  for (unsigned int eq=stokes_neq; eq<neq; ++eq)
    this->sideSetEquations[eq].push_back(basalSideName);
}

Teuchos::Array< Teuchos::RCP<const PHX::FieldTag> >
LandIce::StokesFOHydrology::
buildEvaluators(
  PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
  const Albany::MeshSpecsStruct& meshSpecs,
  Albany::StateManager& stateMgr,
  Albany::FieldManagerChoice fmchoice,
  const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  // Call constructeEvaluators<EvalT>(*rfm[0], *meshSpecs[0], stateMgr);
  // for each EvalT in PHAL::AlbanyTraits::BEvalTypes
  Albany::ConstructEvaluatorsOp<StokesFOHydrology> op(
    *this, fm0, meshSpecs, stateMgr, fmchoice, responseList);
  Sacado::mpl::for_each<PHAL::AlbanyTraits::BEvalTypes> fe(op);
  return *op.tags;
}

void StokesFOHydrology::
constructDirichletEvaluators(const Albany::MeshSpecsStruct& meshSpecs)
{
  // Construct Dirichlet evaluators for all nodesets and names
  std::vector<std::string> dir_names(stokes_neq+hydro_neq);
  for (int i=0; i<stokes_neq; i++) {
    std::stringstream s; s << "U" << i;
    dir_names[i] = s.str();
  }
  for (int i=0; i<hydro_neq; ++i) {
    dir_names[stokes_neq+i] = hydro_dofs_names[i];
  }

  Albany::BCUtils<Albany::DirichletTraits> dirUtils;
  dfm = dirUtils.constructBCEvaluators(meshSpecs.nsNames, dir_names, this->params, this->paramLib, neq);
  use_sdbcs_ = dirUtils.useSDBCs(); 
  offsets_ = dirUtils.getOffsets();
  nodeSetIDs_ = dirUtils.getNodeSetIDs();
}

// Neumann BCs
void StokesFOHydrology::
constructNeumannEvaluators (const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs)
{
  Albany::BCUtils<Albany::NeumannTraits> nbcUtils;

  // Check to make sure that Neumann BCs are given in the input file
  if(!nbcUtils.haveBCSpecified(this->params)) {
     return;
  }

  // Construct BC evaluators for all side sets and names
  // Note that the string index sets up the equation offset, so ordering is important
  // Also, note that we only have neumann conditions for the ice. Hydrology can also
  // have neumann BC, but they are homogeneous (do-nothing).

  // Stokes BCs
  std::vector<std::string> stokes_neumann_names(stokes_neq + 1);
  Teuchos::Array<Teuchos::Array<int> > stokes_offsets;
  stokes_offsets.resize(stokes_neq + 1);

  stokes_neumann_names[0] = "U0";
  stokes_offsets[0].resize(1);
  stokes_offsets[0][0] = 0;
  stokes_offsets[stokes_neq].resize(stokes_neq);
  stokes_offsets[stokes_neq][0] = 0;

  if (neq>1)
  {
    stokes_neumann_names[1] = "U1";
    stokes_offsets[1].resize(1);
    stokes_offsets[1][0] = 1;
    stokes_offsets[stokes_neq][1] = 1;
  }

  stokes_neumann_names[stokes_neq] = "all";

  std::vector<std::string> stokes_cond_names(1);
  stokes_cond_names[0] = "lateral";

  nfm.resize(1); // LandIce problem only has one element block

  nfm[0] = nbcUtils.constructBCEvaluators(meshSpecs, stokes_neumann_names, stokes_dofs_names, true, 0,
                                          stokes_cond_names, stokes_offsets, dl,
                                          this->params, this->paramLib);
}

Teuchos::RCP<const Teuchos::ParameterList>
StokesFOHydrology::getValidProblemParameters () const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL = StokesFOBase::getStokesFOBaseProblemParameters();

  validPL->sublist("LandIce Hydrology", false, "");
  validPL->sublist("LandIce Field Norm", false, "");
  validPL->sublist("LandIce Viscosity", false, "");
  validPL->sublist("LandIce Physical Parameters", false, "");
  validPL->sublist("LandIce Basal Friction Coefficient", false, "Parameters needed to compute the basal friction coefficient");

  return validPL;
}

void StokesFOHydrology::setFieldsProperties () {
  StokesFOBase::setFieldsProperties();

  // Set dof's properties
  setSingleFieldProperties(var_names.water_pressure, FRT::Scalar, FST::Scalar, FL::Node);
  setSingleFieldProperties(var_names.water_thickness, FRT::Scalar, FST::Scalar, FL::Node);
  setSingleFieldProperties(var_names.till_water_storage, FRT::Scalar, FST::Scalar, FL::Node);

  setSingleFieldProperties(effective_pressure_name, FRT::Scalar, FST::Scalar, FL::Node);
  setSingleFieldProperties(var_names.water_discharge, FRT::Gradient, FST::Scalar, FL::QuadPoint);
  setSingleFieldProperties(var_names.hydraulic_potential, FRT::Scalar, FST::Scalar, FL::Node);
  setSingleFieldProperties(var_names.surface_water_input, FRT::Scalar, FST::ParamScalar, FL::Node);

  is_ss_computed_field[basalSideName][effective_pressure_name] = true;
}

void StokesFOHydrology::setupEvaluatorRequests () {
  StokesFOBase::setupEvaluatorRequests();

  ss_build_interp_ev[basalSideName][var_names.water_pressure][InterpolationRequest::QP_VAL] = true; 
  if (!eliminate_h) {
    // If we eliminate h, then we compute water thickness, rather than interpolate the dof
    ss_build_interp_ev[basalSideName][var_names.water_thickness][InterpolationRequest::QP_VAL] = true; 
  }
  ss_build_interp_ev[basalSideName][var_names.hydraulic_potential][InterpolationRequest::GRAD_QP_VAL] = true; 
  ss_build_interp_ev[basalSideName][var_names.water_discharge][InterpolationRequest::CELL_VAL] = true; 
  ss_build_interp_ev[basalSideName][var_names.surface_water_input][InterpolationRequest::QP_VAL] = true; 
}

} // namespace LandIce

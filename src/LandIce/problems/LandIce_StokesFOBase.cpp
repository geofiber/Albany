#include "LandIce_StokesFOBase.hpp"
#include "Teuchos_CompilerCodeTweakMacros.hpp"

#include <string.hpp> // For util::upper_case (do not confuse this with <string>! string.hpp is an Albany file)

namespace LandIce {

StokesFOBase::
StokesFOBase (const Teuchos::RCP<Teuchos::ParameterList>& params_,
                        const Teuchos::RCP<Teuchos::ParameterList>& discParams_,
                        const Teuchos::RCP<ParamLib>& paramLib_,
                        const int numDim_)
 : Albany::AbstractProblem(params_, paramLib_, numDim_)
 , discParams (discParams_)
 , numDim(numDim_)
 , use_sdbcs_(false)
{
  // Need to allocate a fields in mesh database
  if (params->isParameter("Required Fields"))
  {
    // Need to allocate a fields in mesh database
    Teuchos::Array<std::string> req = params->get<Teuchos::Array<std::string> > ("Required Fields");
    for (int i(0); i<req.size(); ++i)
      this->requirements.push_back(req[i]);
  }

  // Parsing the LandIce boundary conditions sublist
  auto landice_bcs_params = Teuchos::sublist(params,"LandIce BCs");
  int num_bcs = landice_bcs_params->get<int>("Number",0);
  for (int i=0; i<num_bcs; ++i) {
    auto this_bc = Teuchos::sublist(landice_bcs_params,Albany::strint("BC",i));
    std::string type_str = util::upper_case(this_bc->get<std::string>("Type"));

    LandIceBC type;
    if (type_str=="BASAL FRICTION") {
      type = LandIceBC::BasalFriction;
    } else if (type_str=="LATERAL") {
      type = LandIceBC::Lateral;
    } else if (type_str=="SYNTETIC TEST") {
      type = LandIceBC::SynteticTest;
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameterValue,
                                  "Error! Unknown LandIce bc '" + type_str + "'.\n");
    }
    landice_bcs[type].push_back(this_bc);
  }

  // Surface side, where velocity diagnostics are computed (e.g., velocity mismatch)
  surfaceSideName = params->isParameter("Surface Side Name") ? params->get<std::string>("Surface Side Name") : INVALID_STR;

  // Basal side, where thickness-related diagnostics are computed (e.g., SMB)
  basalSideName = params->isParameter("Basal Side Name") ? params->get<std::string>("Basal Side Name") : INVALID_STR;

  if (params->sublist("LandIce Physical Parameters").isParameter("Clausius-Clapeyron Coefficient") &&
      params->sublist("LandIce Physical Parameters").get<double>("Clausius-Clapeyron Coefficient")!=0.0) {
    viscosity_use_corrected_temperature = true;
  } else {
    viscosity_use_corrected_temperature = false;
  }
  compute_dissipation = params->sublist("LandIce Viscosity").get("Extract Strain Rate Sq", false);

  // Setup velocity dof and resid names. Derived classes should _append_ to these
  dof_names.resize(1);
  resid_names.resize(1);
  scatter_names.resize(1);

  dof_names[0] = "Velocity";
  resid_names[0] = dof_names[0] + " Residual";
  scatter_names[0] = "Scatter " + resid_names[0];

  dof_offsets.resize(1);
  dof_offsets[0] = 0;
  vecDimFO = std::min((int)neq,(int)2);

  // Names of some common fields. User can set them in the problem section, in case they are
  // loaded from mesh, where they are saved with a different name
  surface_height_name         = params->sublist("Variables Names").get<std::string>("Surface Height Name","surface_height");
  ice_thickness_name          = params->sublist("Variables Names").get<std::string>("Ice Thickness Name" ,"ice_thickness");
  temperature_name            = params->sublist("Variables Names").get<std::string>("Temperature Name"   ,"temperature");
  corrected_temperature_name  = "corrected_" + temperature_name;
  bed_topography_name         = params->sublist("Variables Names").get<std::string>("Bed Topography Name"    ,"bed_topography");
  flow_factor_name            = params->sublist("Variables Names").get<std::string>("Flow Factor Name"       ,"flow_factor");
  stiffening_factor_name      = params->sublist("Variables Names").get<std::string>("Stiffening Factor Name" ,"stiffening_factor");
  effective_pressure_name     = params->sublist("Variables Names").get<std::string>("Effective Pressure Name","effective_pressure");
  vertically_averaged_velocity_name = params->sublist("Variables Names").get<std::string>("Vertically Averaged Velocity Name","vertically_averaged_velocity");

  // Mark the velocity as computed
  is_computed_field[dof_names[0]] = true;

  // By default, we are not coupled to any other physics
  temperature_coupled = false;
  hydrology_coupled = false;
}

void StokesFOBase::buildProblem (Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
                                 Albany::StateManager& stateMgr)
{
  using Teuchos::rcp;

  // Building cell basis and cubature
  const CellTopologyData * const cell_top = &meshSpecs[0]->ctd;
  cellBasis = Albany::getIntrepid2Basis(*cell_top);
  cellType = rcp(new shards::CellTopology (cell_top));

  Intrepid2::DefaultCubatureFactory cubFactory;
  cellCubature = cubFactory.create<PHX::Device, RealType, RealType>(*cellType, meshSpecs[0]->cubatureDegree);

  const int worksetSize     = meshSpecs[0]->worksetSize;
  const int numCellSides    = cellType->getSideCount();
  const int numCellVertices = cellType->getNodeCount();
  const int numCellNodes    = cellBasis->getCardinality();
  const int numCellQPs      = cellCubature->getNumPoints();

  dl = rcp(new Albany::Layouts(worksetSize,numCellVertices,numCellNodes,numCellQPs,numDim,vecDimFO));

  int sideDim = numDim-1;
  for (auto it : landice_bcs) {
    for (auto pl: it.second) {
      std::string ssName = pl->get<std::string>("Side Set Name");
      TEUCHOS_TEST_FOR_EXCEPTION (meshSpecs[0]->sideSetMeshSpecs.find(ssName)==meshSpecs[0]->sideSetMeshSpecs.end(), std::logic_error,
                                  "Error! Either the side set name is wrong or something went wrong while building the side mesh specs.\n");
      const Albany::MeshSpecsStruct& sideMeshSpecs = *meshSpecs[0]->sideSetMeshSpecs.at(ssName)[0];

      // Building also side structures
      const CellTopologyData * const side_top = &sideMeshSpecs.ctd;
      sideBasis[ssName] = Albany::getIntrepid2Basis(*side_top);
      sideType[ssName] = rcp(new shards::CellTopology (side_top));

      // If there's no side discretiation, then sideMeshSpecs.cubatureDegree will be -1, and the user need to specify a cubature degree somewhere else
      int sideCubDegree = sideMeshSpecs.cubatureDegree;
      if (pl->isParameter("Cubature Degree")) {
        sideCubDegree = pl->get<int>("Cubature Degree");
      }
      TEUCHOS_TEST_FOR_EXCEPTION (sideCubDegree<0, std::runtime_error, "Error! Missing cubature degree information on side '" << ssName << ".\n"
                                                                       "       Either add a side discretization, or specify 'Cubature Degree' in sublist '" + pl->name() + "'.\n");
      sideCubature[ssName] = cubFactory.create<PHX::Device, RealType, RealType>(*sideType[ssName], sideCubDegree);

      int numSideVertices = sideType[ssName]->getNodeCount();
      int numSideNodes    = sideBasis[ssName]->getCardinality();
      int numSideQPs      = sideCubature[ssName]->getNumPoints();

      dl->side_layouts[ssName] = rcp(new Albany::Layouts(worksetSize,numSideVertices,numSideNodes,
                                                         numSideQPs,sideDim,numDim,numCellSides,vecDimFO));
    }
  }

  // If we have velocity diagnostics, we need surface side stuff
  if (!isInvalid(surfaceSideName) && dl->side_layouts.find(surfaceSideName)==dl->side_layouts.end())
  {
    TEUCHOS_TEST_FOR_EXCEPTION (meshSpecs[0]->sideSetMeshSpecs.find(surfaceSideName)==meshSpecs[0]->sideSetMeshSpecs.end(), std::logic_error,
                                  "Error! Either 'Surface Side Name' is wrong or something went wrong while building the side mesh specs.\n");

    const Albany::MeshSpecsStruct& surfaceMeshSpecs = *meshSpecs[0]->sideSetMeshSpecs.at(surfaceSideName)[0];

    // Building also surface side structures
    const CellTopologyData * const side_top = &surfaceMeshSpecs.ctd;
    sideBasis[surfaceSideName] = Albany::getIntrepid2Basis(*side_top);
    sideType[surfaceSideName]= rcp(new shards::CellTopology (side_top));

    sideCubature[surfaceSideName] = cubFactory.create<PHX::Device, RealType, RealType>(*sideType[surfaceSideName], surfaceMeshSpecs.cubatureDegree);

    int numSurfaceSideVertices = sideType[surfaceSideName]->getNodeCount();
    int numSurfaceSideNodes    = sideBasis[surfaceSideName]->getCardinality();
    int numSurfaceSideQPs      = sideCubature[surfaceSideName]->getNumPoints();

    dl->side_layouts[surfaceSideName] = rcp(new Albany::Layouts(worksetSize,numSurfaceSideVertices,numSurfaceSideNodes,
                                                                numSurfaceSideQPs,sideDim,numDim,numCellSides,vecDimFO));
  }

  // If we have thickness or surface velocity diagnostics, we may need basal side stuff
  if (!isInvalid(basalSideName) && dl->side_layouts.find(basalSideName)==dl->side_layouts.end())
  {
    TEUCHOS_TEST_FOR_EXCEPTION (meshSpecs[0]->sideSetMeshSpecs.find(basalSideName)==meshSpecs[0]->sideSetMeshSpecs.end(), std::logic_error,
                                  "Error! Either 'Basal Side Name' is wrong or something went wrong while building the side mesh specs.\n");

    const Albany::MeshSpecsStruct& basalMeshSpecs = *meshSpecs[0]->sideSetMeshSpecs.at(basalSideName)[0];

    // Building also basal side structures
    const CellTopologyData * const side_top = &basalMeshSpecs.ctd;
    sideBasis[basalSideName] = Albany::getIntrepid2Basis(*side_top);
    sideType[basalSideName]= rcp(new shards::CellTopology (side_top));

    sideCubature[basalSideName] = cubFactory.create<PHX::Device, RealType, RealType>(*sideType[basalSideName], basalMeshSpecs.cubatureDegree);

    int numbasalSideVertices = sideType[basalSideName]->getNodeCount();
    int numbasalSideNodes    = sideBasis[basalSideName]->getCardinality();
    int numbasalSideQPs      = sideCubature[basalSideName]->getNumPoints();

    dl->side_layouts[basalSideName] = rcp(new Albany::Layouts(worksetSize,numbasalSideVertices,numbasalSideNodes,
                                                              numbasalSideQPs,sideDim,numDim,numCellSides,vecDimFO));
  }

#ifdef OUTPUT_TO_SCREEN
  Teuchos::RCP<Teuchos::FancyOStream> out = Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout));
  int commRank = Teuchos::GlobalMPISession::getRank();
  int commSize = Teuchos::GlobalMPISession::getNProc();
  out->setProcRankAndSize(commRank, commSize);
  out->setOutputToRootOnly(0);

  *out << "=== Field Dimensions ===\n"
       << " Volume:\n"
       << "   Workset     = " << worksetSize << "\n"
       << "   Vertices    = " << numCellVertices << "\n"
       << "   CellNodes   = " << numCellNodes << "\n"
       << "   CellQuadPts = " << numCellQPs << "\n"
       << "   Dim         = " << numDim << "\n"
       << "   VecDim      = " << neq << "\n"
       << "   VecDimFO    = " << vecDimFO << "\n";
  for (auto it : dl_side) {
    *out << " Side Set '" << it.first << "':\n" 
         << "  Vertices   = " << it.second->vertices_vector->dimension(1) << "\n"
         << "  Nodes      = " << it.second->node_scalar->dimension(1) << "\n"
         << "  QuadPts    = " << it.second->qp_scalar->dimension(1) << "\n";
  }
#endif

  // Parse the input/output fields properties
  // We do this BEFORE the evaluators request/construction,
  // so we have all the info about the input fields.
  parseInputFields ();

  // Set the scalar type of all the fields, plus, if needed, their rank or whether they are computed.
  setFieldsProperties ();

  // Prepare the requests of interpolation/utility evaluators
  setupEvaluatorRequests ();

  /* Construct All Phalanx Evaluators */
  TEUCHOS_TEST_FOR_EXCEPTION(meshSpecs.size()!=1,std::logic_error,"Problem supports one Material Block");
  fm.resize(1);
  fm[0]  = rcp(new PHX::FieldManager<PHAL::AlbanyTraits>);
  buildEvaluators(*fm[0], *meshSpecs[0], stateMgr, Albany::BUILD_RESID_FM,Teuchos::null);

  // Build a dirichlet fm if nodesets are present
  if (meshSpecs[0]->nsNames.size() >0) {
    constructDirichletEvaluators(*meshSpecs[0]);
  }

  // Build a neumann fm if sidesets are present
  if(meshSpecs[0]->ssNames.size() > 0) {
     constructNeumannEvaluators(meshSpecs[0]);
  }
}

Teuchos::RCP<Teuchos::ParameterList>
StokesFOBase::getStokesFOBaseProblemParameters () const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL = this->getGenericProblemParams("ValidStokesFOProblemParams");

  validPL->set<bool> ("Extruded Column Coupled in 2D Response", false, "Boolean describing whether the extruded column is coupled in 2D response");
  validPL->sublist("Stereographic Map", false, "");
  validPL->sublist("LandIce BCs", false, "Specify boundary conditions specific to LandIce (bypass usual Neumann/Dirichlet classes)");
  validPL->sublist("LandIce Viscosity", false, "");
  validPL->sublist("LandIce Effective Pressure Surrogate", false, "Parameters needed to compute the effective pressure surrogate");
  validPL->sublist("LandIce L2 Projected Boundary Laplacian", false, "Parameters needed to compute the L2 Projected Boundary Laplacian");
  validPL->sublist("LandIce Surface Gradient", false, "");
  validPL->sublist("Variables Names",false,"Sublist where we can specify a user-defined name for variables.");
  validPL->set<std::string> ("Basal Side Name", "", "Name of the basal side set");
  validPL->set<std::string> ("Surface Side Name", "", "Name of the surface side set");
  validPL->sublist("Body Force", false, "");
  validPL->sublist("LandIce Field Norm", false, "");
  validPL->sublist("LandIce Physical Parameters", false, "");
  validPL->sublist("LandIce Noise", false, "");
  validPL->set<bool>("Use Time Parameter", false, "Solely to use Solver Method = Continuation");
  validPL->set<bool>("Print Stress Tensor", false, "Whether to save stress tensor in the mesh");

  return validPL;
}

void StokesFOBase::parseInputFields ()
{
  std::string stateName, fieldName, param_name;

  // Getting the names of the distributed parameters (they won't have to be loaded as states)
  if (this->params->isSublist("Distributed Parameters")) {
    Teuchos::ParameterList& dist_params_list =  this->params->sublist("Distributed Parameters");
    Teuchos::ParameterList* param_list;
    int numParams = dist_params_list.get<int>("Number of Parameter Vectors",0);
    for (int p_index=0; p_index< numParams; ++p_index) {
      std::string parameter_sublist_name = Albany::strint("Distributed Parameter", p_index);
      if (dist_params_list.isSublist(parameter_sublist_name)) {
        // The better way to specify dist params: with sublists
        param_list = &dist_params_list.sublist(parameter_sublist_name);
        param_name = param_list->get<std::string>("Name");
        dist_params_name_to_mesh_part[param_name] = param_list->get<std::string>("Mesh Part","");
        is_extruded_param[param_name] = param_list->get<bool>("Extruded",false);
        int extruded_param_level = param_list->get<int>("Extruded Param Level",0);
        extruded_params_levels.insert(std::make_pair(param_name, extruded_param_level));
        save_sensitivities[param_name]=param_list->get<bool>("Save Sensitivity",false);
      } else {
        // Legacy way to specify dist params: with parameter entries. Note: no mesh part can be specified.
        param_name = dist_params_list.get<std::string>(Albany::strint("Parameter", p_index));
        dist_params_name_to_mesh_part[param_name] = "";
      }
      is_dist_param[param_name] = true;
      is_input_field[param_name] = true;
      is_dist[param_name] = true;
      is_dist[param_name+"_upperbound"] = true;
      dist_params_name_to_mesh_part[param_name+"_upperbound"] = dist_params_name_to_mesh_part[param_name];
      is_dist[param_name+"_lowerbound"] = true;
      dist_params_name_to_mesh_part[param_name+"_lowerbound"] = dist_params_name_to_mesh_part[param_name];
    }
  }

  //Dirichlet fields need to be distributed but they are not necessarily parameters.
  if (this->params->isSublist("Dirichlet BCs")) {
    Teuchos::ParameterList dirichlet_list = this->params->sublist("Dirichlet BCs");
    for(auto it = dirichlet_list.begin(); it !=dirichlet_list.end(); ++it) {
      std::string pname = dirichlet_list.name(it);
      if(dirichlet_list.isParameter(pname) && dirichlet_list.isType<std::string>(pname)){ //need to check, because pname could be the name sublist
        is_dist[dirichlet_list.get<std::string>(pname)]=true;
        dist_params_name_to_mesh_part[dirichlet_list.get<std::string>(pname)]="";
      }
    }
  }

  // Volume mesh requirements
  Teuchos::ParameterList& req_fields_info = discParams->sublist("Required Fields Info");
  int num_fields = req_fields_info.get<int>("Number Of Fields",0);

  std::string fieldType, fieldUsage, meshPart;
  bool nodal_state, scalar_state;
  for (int ifield=0; ifield<num_fields; ++ifield) {
    Teuchos::ParameterList& thisFieldList = req_fields_info.sublist(Albany::strint("Field", ifield));

    // Get current state specs
    fieldName = thisFieldList.get<std::string>("Field Name");
    stateName = thisFieldList.get<std::string>("State Name", fieldName);
    fieldUsage = thisFieldList.get<std::string>("Field Usage","Input"); // WARNING: assuming Input if not specified

    if (fieldUsage == "Unused") {
      continue;
    }

    fieldType  = thisFieldList.get<std::string>("Field Type");

    is_dist_param.insert(std::pair<std::string,bool>(stateName, false));  //gets inserted only if not there.
    is_dist.insert(std::pair<std::string,bool>(stateName, false));        //gets inserted only if not there.

    meshPart = is_dist[stateName] ? dist_params_name_to_mesh_part[stateName] : "";

    if(fieldType == "Elem Scalar") {
      nodal_state = false;
      scalar_state = true;
    } else if(fieldType == "Node Scalar") {
      nodal_state = true;
      scalar_state = true;
    } else if(fieldType == "Elem Vector") {
      nodal_state = false;
      scalar_state = false;
    } else if(fieldType == "Node Vector") {
      nodal_state = true;
      scalar_state = false;
    }

    // Do we need to load/gather the state/parameter?
    if (is_dist_param[stateName] || fieldUsage == "Input" || fieldUsage == "Input-Output") {
      // A parameter to gather or a field to load
      is_input_field[stateName] = true;
    }

    // Set rank, location and scalar type.
    // Note: output fields *may* have ScalarType = Scalar. This may seem like a problem,
    //       but it's not, since the Save(SideSet)StateField evaluators are only created
    //       for the Residual.
    field_rank[stateName] = scalar_state ? 0 : 1;
    field_location[stateName] = nodal_state ? FieldLocation::Node : FieldLocation::Cell;
    field_scalar_type[stateName] |= is_dist_param[stateName] ? FieldScalarType::ParamScalar : FieldScalarType::Real;
  }

  // Side set requirements
  Teuchos::Array<std::string> ss_names;
  if (discParams->sublist("Side Set Discretizations").isParameter("Side Sets")) {
    ss_names = discParams->sublist("Side Set Discretizations").get<Teuchos::Array<std::string>>("Side Sets");
  } 
  for (int i=0; i<ss_names.size(); ++i) {
    const std::string& ss_name = ss_names[i];
    Teuchos::ParameterList& info = discParams->sublist("Side Set Discretizations").sublist(ss_name).sublist("Required Fields Info");
    num_fields = info.get<int>("Number Of Fields",0);

    Teuchos::RCP<Albany::Layouts> ss_dl = dl->side_layouts.at(ss_name);
    for (int ifield=0; ifield<num_fields; ++ifield) {
      Teuchos::ParameterList& thisFieldList =  info.sublist(Albany::strint("Field", ifield));

      // Get current state specs
      fieldName = thisFieldList.get<std::string>("Field Name");
      stateName = thisFieldList.get<std::string>("State Name", fieldName);
      fieldName = fieldName + "_" + ss_name;
      fieldUsage = thisFieldList.get<std::string>("Field Usage","Input"); // WARNING: assuming Input if not specified

      if (fieldUsage == "Unused") {
        continue;
      }

      meshPart = ""; // Distributed parameters are defined either on the whole volume mesh or on a whole side mesh. Either way, here we want "" as part (the whole mesh).

      fieldType  = thisFieldList.get<std::string>("Field Type");

      if(fieldType == "Elem Scalar") {
        nodal_state = false;
        scalar_state = true;
      } else if(fieldType == "Node Scalar") {
        nodal_state = true;
        scalar_state = true;
      } else if(fieldType == "Elem Vector") {
        nodal_state = false;
        scalar_state = false;
      } else if(fieldType == "Node Vector") {
        nodal_state = true;
        scalar_state = false;
      } else if(fieldType == "Elem Layered Scalar") {
        nodal_state = false;
        scalar_state = true;
      } else if(fieldType == "Node Layered Scalar") {
        nodal_state = true;
        scalar_state = true;
      } else if(fieldType == "Elem Layered Vector") {
        nodal_state = false;
        scalar_state = false;
      } else if(fieldType == "Node Layered Vector") {
        nodal_state = true;
        scalar_state = false;
      }

      if (!is_dist_param[stateName] && (fieldUsage == "Input" || fieldUsage == "Input-Output")) {
        // A parameter to gather or a field to load
        is_ss_input_field[ss_name][stateName] = true;
      }

      field_rank[stateName] = scalar_state ? 0 : 1;
      ss_field_location[ss_name][stateName] = nodal_state ? FieldLocation::Node : FieldLocation::Cell;
    }
  }
}

void StokesFOBase::setFieldsProperties ()
{
  // All dofs have scalar type Scalar (i.e., they depend on the solution)
  // and are computed.
  for (auto it : dof_names) {
    field_scalar_type[it] = FieldScalarType::Scalar;
    is_computed_field[it] = true;
  }

  // All volume input fields are Real or ParamScalar (if they are dist params)
  for (auto it : is_input_field) {
    field_scalar_type[it.first] = is_dist_param[it.first] ? FieldScalarType::ParamScalar : FieldScalarType::Real;
  }

  // All side inputs are Real (there are no side dist params)
  for (auto it : is_ss_input_field) {
    for (auto it2 : it.second) {
      field_scalar_type[it2.first] = FieldScalarType::Real;
    }
  }

  // Set scalar types of known fields. For ice_thickness and surface_height, the scalar type should be at least MeshScalar.
  // If things are different in derived classes, then adjust.
  field_scalar_type[ice_thickness_name] |= FieldScalarType::MeshScalar;   // Note: use |= operator, so we keep the strongest.
  field_scalar_type[surface_height_name] |= FieldScalarType::MeshScalar;  //       If they are dist param, the st will remain ParamScalar
  field_scalar_type[vertically_averaged_velocity_name] = FieldScalarType::Scalar;
  field_scalar_type[corrected_temperature_name] = field_scalar_type[temperature_name] |    // Note: corr temp is built from temp and surf height. Combine their scalar types.
                                                  field_scalar_type[surface_height_name];  //       If derived class changes the type of temp or surf height, need to adjust this too.
  field_scalar_type[bed_topography_name] |= FieldScalarType::MeshScalar;

  // Set ranks of known fields
  // Note: we only care about fields that MAY not be parsed among the inputs.
  field_rank[dof_names[0]] = 1;
  field_rank[ice_thickness_name]   = 0;
  field_rank[surface_height_name]  = 0;
  field_rank[effective_pressure_name] = 0;
  field_rank[vertically_averaged_velocity_name]  = 1;
  field_rank[temperature_name] = 0;
  field_rank[corrected_temperature_name] = 0;
  field_rank[stiffening_factor_name] = 0;
}

void StokesFOBase::requestInterpolationEvaluator (
    const std::string& fname,
    // const int rank,
    const FieldLocation location,
    // const FieldScalarType scalar_type,
    const InterpolationRequest request)
{
  // TEUCHOS_TEST_FOR_EXCEPTION (field_rank.find(fname)!=field_rank.end() && field_rank[fname]!=rank, std::logic_error, 
  //                             "Error! Attempt to mark field '" + fname + " with rank " + std::to_string(rank) +
  //                             ", when it was previously marked as of rank " + std::to_string(field_rank[fname]) + ".\n");
  TEUCHOS_TEST_FOR_EXCEPTION (field_location.find(fname)!=field_location.end() && field_location[fname]!=location, std::logic_error, 
                              "Error! Attempt to mark field '" + fname + " as located at " + e2str(location) +
                              ", when it was previously marked as located at " + e2str(field_location[fname]) + ".\n");

  build_interp_ev[fname][request] = true;
  // field_rank[fname] = rank;
  field_location[fname] = location;
  // field_scalar_type[fname] |= scalar_type;
}

void StokesFOBase::requestSideSetInterpolationEvaluator (
    const std::string& ss_name,
    const std::string& fname,
    // const int rank,
    const FieldLocation location,
    // const FieldScalarType scalar_type,
    const InterpolationRequest request)
{
  // TEUCHOS_TEST_FOR_EXCEPTION (field_rank.find(fname)!=field_rank.end() && field_rank[fname]!=rank, std::logic_error, 
                              // "Error! Attempt to mark field '" + fname + " with rank " + std::to_string(rank) +
                              // ", when it was previously marked as of rank " + std::to_string(field_rank[fname]) + ".\n");
  TEUCHOS_TEST_FOR_EXCEPTION (ss_field_location[ss_name].find(fname)!=ss_field_location[ss_name].end() && ss_field_location[ss_name][fname]!=location, std::logic_error, 
                              "Error! Attempt to mark field '" + fname + " as located at " + e2str(location) +
                              ", when it was previously marked as located at " + e2str(ss_field_location[ss_name][fname]) + ".\n");

  ss_build_interp_ev[ss_name][fname][request] = true;
  // field_rank[fname] = rank;
  ss_field_location[ss_name][fname] = location;
  // field_scalar_type[fname] |= scalar_type;
}

void StokesFOBase::setupEvaluatorRequests ()
{
  // Volume required interpolations
  requestInterpolationEvaluator(dof_names[0], FieldLocation::Node, InterpolationRequest::QP_VAL); 
  requestInterpolationEvaluator(dof_names[0], FieldLocation::Node, InterpolationRequest::GRAD_QP_VAL); 
  requestInterpolationEvaluator(surface_height_name, FieldLocation::Node, InterpolationRequest::QP_VAL); 
#ifndef CISM_HAS_LANDICE
  // If not coupled with cism, we may have to compute the surface gradient ourselves
  requestInterpolationEvaluator(surface_height_name, FieldLocation::Node, InterpolationRequest::GRAD_QP_VAL); 
#endif
  if (is_input_field[temperature_name]) {
    requestInterpolationEvaluator(temperature_name, FieldLocation::Node, InterpolationRequest::CELL_VAL); 
  }
  if (is_input_field[stiffening_factor_name]) {
    requestInterpolationEvaluator(stiffening_factor_name, FieldLocation::Node, InterpolationRequest::QP_VAL); 
  }

  // Basal Friction BC requests
  for (auto it : landice_bcs[LandIceBC::BasalFriction]) {
    std::string ssName = it->get<std::string>("Side Set Name");
    // BasalFriction BC needs velocity on the side, and BFs/coords
    // And if we compute grad beta, we may even need the velocity gradient and the effective pressure gradient
    // (which, if effevtive_pressure is a dist param, needs to be projected to the side)

    ss_utils_needed[ssName][UtilityRequest::BFS] = true;
    ss_utils_needed[ssName][UtilityRequest::QP_COORDS] = true;  // Only really needed if stereographic map is used.

    requestSideSetInterpolationEvaluator(ssName, dof_names[0], FieldLocation::Node, InterpolationRequest::CELL_TO_SIDE); 
    requestSideSetInterpolationEvaluator(ssName, dof_names[0], FieldLocation::Node, InterpolationRequest::QP_VAL); 
    requestSideSetInterpolationEvaluator(ssName, dof_names[0], FieldLocation::Node, InterpolationRequest::GRAD_QP_VAL); 
    requestSideSetInterpolationEvaluator(ssName, effective_pressure_name, FieldLocation::Node, InterpolationRequest::GRAD_QP_VAL); 
    requestSideSetInterpolationEvaluator(ssName, effective_pressure_name, FieldLocation::Node, InterpolationRequest::CELL_TO_SIDE); 

    // For "Given Field" and "Exponent of Given Field" we also need to interpolate the given field at the quadrature points
    auto& bfc = it->sublist("Basal Friction Coefficient");
    const auto type = util::upper_case(bfc.get<std::string>("Type"));
    if (type=="GIVEN FIELD" || type=="EXPONENT OF GIVEN FIELD") {
      requestSideSetInterpolationEvaluator(ssName, bfc.get<std::string>("Given Field Variable Name"),
                                           FieldLocation::Node, InterpolationRequest::QP_VAL);
      requestSideSetInterpolationEvaluator(ssName, bfc.get<std::string>("Given Field Variable Name"),
                                           FieldLocation::Node, InterpolationRequest::CELL_TO_SIDE);
    }

    // If zero on floating, we also need bed topography and thickness
    if (bfc.get<bool>("Zero Beta On Floating Ice", false)) {
      requestSideSetInterpolationEvaluator(ssName, bed_topography_name, FieldLocation::Node, InterpolationRequest::QP_VAL);
      requestSideSetInterpolationEvaluator(ssName, bed_topography_name, FieldLocation::Node, InterpolationRequest::CELL_TO_SIDE);

      requestSideSetInterpolationEvaluator(ssName, ice_thickness_name, FieldLocation::Node, InterpolationRequest::QP_VAL);
      requestSideSetInterpolationEvaluator(ssName, ice_thickness_name, FieldLocation::Node, InterpolationRequest::CELL_TO_SIDE);
    }
  }

  // Lateral BC requests
  for (auto it : landice_bcs[LandIceBC::Lateral]) {
    std::string ssName = it->get<std::string>("Side Set Name");

    // Lateral bc needs thickness ...
    requestSideSetInterpolationEvaluator(ssName, ice_thickness_name, FieldLocation::Node, InterpolationRequest::CELL_TO_SIDE); 
    requestSideSetInterpolationEvaluator(ssName, ice_thickness_name, FieldLocation::Node, InterpolationRequest::QP_VAL); 

    // ... possibly surface height ...
    requestSideSetInterpolationEvaluator(ssName, surface_height_name, FieldLocation::Node, InterpolationRequest::CELL_TO_SIDE);
    requestSideSetInterpolationEvaluator(ssName, surface_height_name, FieldLocation::Node, InterpolationRequest::QP_VAL); 

    // ... and BFs (including normals)
    ss_utils_needed[ssName][UtilityRequest::BFS] = true;
    ss_utils_needed[ssName][UtilityRequest::NORMALS] = true;
    ss_utils_needed[ssName][UtilityRequest::QP_COORDS] = true;
  }

  // Surface diagnostics
  if (!isInvalid(surfaceSideName)) {
    // Surface velocity diagnostic requires dof at qps on surface side
    requestSideSetInterpolationEvaluator(surfaceSideName, dof_names[0], FieldLocation::Node, InterpolationRequest::CELL_TO_SIDE); 
    requestSideSetInterpolationEvaluator(surfaceSideName, dof_names[0], FieldLocation::Node, InterpolationRequest::QP_VAL); 

    // ... and observed surface velocity ...
    // NOTE: RMS could be either scalar or vector. The states registration should have figure it out (if the field is listed as input).
    if (is_ss_input_field[surfaceSideName]["observed_surface_velocity"]) {
      requestSideSetInterpolationEvaluator(surfaceSideName, "observed_surface_velocity", FieldLocation::Node, InterpolationRequest::QP_VAL);
    }
    if (is_ss_input_field[surfaceSideName]["observed_surface_velocity_RMS"]) {
      requestSideSetInterpolationEvaluator(surfaceSideName, "observed_surface_velocity_RMS", FieldLocation::Node, InterpolationRequest::QP_VAL); 
    }

    // ... and BFs
    ss_utils_needed[surfaceSideName][UtilityRequest::BFS] = true;

    if (!isInvalid(basalSideName) && is_input_field[stiffening_factor_name]) {
      // Surface velocity diagnostics *may* add a basal side regularization
      requestSideSetInterpolationEvaluator(basalSideName, stiffening_factor_name, FieldLocation::Node, InterpolationRequest::CELL_TO_SIDE); 
      requestSideSetInterpolationEvaluator(basalSideName, stiffening_factor_name, FieldLocation::Node, InterpolationRequest::QP_VAL); 
      requestSideSetInterpolationEvaluator(basalSideName, stiffening_factor_name, FieldLocation::Node, InterpolationRequest::GRAD_QP_VAL); 

      ss_utils_needed[basalSideName][UtilityRequest::BFS] = true;
    }
  }

  // SMB-related diagnostics
  if (!isInvalid(basalSideName)) {
    // Needs BFs
    ss_utils_needed[basalSideName][UtilityRequest::BFS] = true;

    // SMB evaluators may need velocity, averaged velocity, and thickness
    requestSideSetInterpolationEvaluator(basalSideName, dof_names[0], FieldLocation::Node, InterpolationRequest::CELL_TO_SIDE); 
    requestSideSetInterpolationEvaluator(basalSideName, dof_names[0], FieldLocation::Node, InterpolationRequest::QP_VAL); 
    requestSideSetInterpolationEvaluator(basalSideName, vertically_averaged_velocity_name, FieldLocation::Node, InterpolationRequest::QP_VAL); 
    requestSideSetInterpolationEvaluator(basalSideName, ice_thickness_name, FieldLocation::Node, InterpolationRequest::QP_VAL); 
    requestSideSetInterpolationEvaluator(basalSideName, ice_thickness_name, FieldLocation::Node, InterpolationRequest::GRAD_QP_VAL); 
    requestSideSetInterpolationEvaluator(basalSideName, ice_thickness_name, FieldLocation::Node, InterpolationRequest::CELL_TO_SIDE); 
    if (is_ss_input_field[basalSideName]["surface_mass_balance"]) {
      requestSideSetInterpolationEvaluator(basalSideName, "surface_mass_balance", FieldLocation::Node, InterpolationRequest::QP_VAL); 
    }
    if (is_ss_input_field[basalSideName]["surface_mass_balance_RMS"]) {
      requestSideSetInterpolationEvaluator(basalSideName, "surface_mass_balance_RMS", FieldLocation::Node, InterpolationRequest::QP_VAL); 
    }
    if (is_ss_input_field[basalSideName]["observed_ice_thickness"]) {
      requestSideSetInterpolationEvaluator(basalSideName, "observed_ice_thickness", FieldLocation::Node, InterpolationRequest::QP_VAL); 
    }
    if (is_ss_input_field[basalSideName]["observed_ice_thickness_RMS"]) {
      requestSideSetInterpolationEvaluator(basalSideName, "observed_ice_thickness_RMS", FieldLocation::Node, InterpolationRequest::QP_VAL); 
    }
  }
}

} // namespace LandIce

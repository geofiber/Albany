//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_ABSTRACT_NODE_FIELD_CONTAINER_HPP
#define ALBANY_ABSTRACT_NODE_FIELD_CONTAINER_HPP

#include "Albany_ThyraTypes.hpp"

#include "Teuchos_RCP.hpp"
#include <map>

namespace Albany {

/*!
 * \brief Abstract interface for an STK NodeField container
 *
 */

class AbstractNodeFieldContainer
{
public:

  AbstractNodeFieldContainer () = default;
  virtual ~AbstractNodeFieldContainer () = default;

  // MV version
  virtual void saveFieldVector(const Teuchos::RCP<const Thyra_MultiVector>& mv, int offset) = 0;
};

typedef std::map<std::string, Teuchos::RCP<AbstractNodeFieldContainer> > NodeFieldContainer;

} // namespace Albany

#endif // ALBANY_ABSTRACT_NODE_FIELD_CONTAINER_HPP

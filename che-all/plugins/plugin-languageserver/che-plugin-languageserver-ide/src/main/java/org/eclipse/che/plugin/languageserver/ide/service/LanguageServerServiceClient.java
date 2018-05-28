/*
 * Copyright (c) 2012-2018 Red Hat, Inc.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors:
 *   Red Hat, Inc. - initial API and implementation
 */
package org.eclipse.che.plugin.languageserver.ide.service;

import static org.eclipse.che.ide.api.jsonrpc.Constants.WS_AGENT_JSON_RPC_ENDPOINT_ID;

import java.util.List;
import java.util.concurrent.TimeoutException;
import javax.inject.Inject;
import javax.inject.Singleton;
import org.eclipse.che.api.core.jsonrpc.commons.JsonRpcError;
import org.eclipse.che.api.core.jsonrpc.commons.JsonRpcException;
import org.eclipse.che.api.core.jsonrpc.commons.RequestTransmitter;
import org.eclipse.che.api.languageserver.shared.model.LanguageRegex;
import org.eclipse.che.api.promises.client.Promise;
import org.eclipse.che.api.promises.client.PromiseError;
import org.eclipse.che.api.promises.client.js.Promises;
import org.eclipse.lsp4j.ServerCapabilities;

@Singleton
public class LanguageServerServiceClient {

  private final RequestTransmitter requestTransmitter;

  @Inject
  public LanguageServerServiceClient(RequestTransmitter requestTransmitter) {
    this.requestTransmitter = requestTransmitter;
  }

  public Promise<ServerCapabilities> initialize(String wsPath) {
    return Promises.create(
        (resolve, reject) ->
            requestTransmitter
                .newRequest()
                .endpointId(WS_AGENT_JSON_RPC_ENDPOINT_ID)
                .methodName("languageServer/initialize")
                .paramsAsString(wsPath)
                .sendAndReceiveResultAsDto(ServerCapabilities.class, 30_000)
                .onSuccess(resolve::apply)
                .onFailure(error -> reject.apply(getPromiseError(error)))
                .onTimeout(() -> reject.apply(getPromiseError())));
  }

  public Promise<List<LanguageRegex>> getLanguageRegexes() {
    return Promises.create(
        (resolve, reject) ->
            requestTransmitter
                .newRequest()
                .endpointId(WS_AGENT_JSON_RPC_ENDPOINT_ID)
                .methodName("languageServer/getLanguageRegexes")
                .noParams()
                .sendAndReceiveResultAsListOfDto(LanguageRegex.class)
                .onSuccess(resolve::apply)
                .onFailure(error -> reject.apply(getPromiseError(error))));
  }

  private PromiseError getPromiseError(JsonRpcError jsonRpcError) {
    return new PromiseError() {
      @Override
      public String getMessage() {
        return jsonRpcError.getMessage();
      }

      @Override
      public Throwable getCause() {
        return new JsonRpcException(jsonRpcError.getCode(), jsonRpcError.getMessage());
      }
    };
  }

  private PromiseError getPromiseError() {
    return new PromiseError() {

      @Override
      public String getMessage() {
        return "Timeout initializing error";
      }

      @Override
      public Throwable getCause() {
        return new TimeoutException();
      }
    };
  }
}
